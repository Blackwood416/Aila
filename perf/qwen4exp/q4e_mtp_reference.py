"""Bounded offline CPU mathematical check for the first four GPU MTP steps.

Not a CPU inference path. Reads one expert at a time and chunks the shared head.
Uses checkpoint shapes, residual_linear_shared/scheme-A semantics, independent
NumPy matmul and causal attention. Preserves the engine's BF16 boundaries:
projection weights round to BF16, fused SwiGLU/MoE dequant weights stay FP32.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def bf(x):
    a=np.asarray(x,dtype=np.float32)
    bits=a.view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)


def read_bf(path):
    return (np.fromfile(path,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)


def sigmoid(x):
    return 1/(1+np.exp(np.clip(-x,-80,80)))


class Checkpoint:
    def __init__(self,directory):
        self.directory=Path(directory)
        self.index=json.loads((self.directory/'model.safetensors.index.json').read_text(encoding='utf-8'))['weight_map']
        self.headers={}

    def tensor(self,name):
        file=self.directory/self.index[name]
        if file not in self.headers:
            with file.open('rb') as f:
                size=struct.unpack('<Q',f.read(8))[0]
                self.headers[file]=(8+size,json.loads(f.read(size)))
        start,header=self.headers[file]; meta=header[name]
        types={'BF16':np.uint16,'F32':np.float32,'U8':np.uint8}
        return np.memmap(file,mode='r',offset=start+meta['data_offsets'][0],dtype=types[meta['dtype']],shape=tuple(meta['shape']))

    def dense(self,name):
        a=self.tensor(name)
        return (a.astype(np.uint32)<<16).view(np.float32) if a.dtype==np.uint16 else np.array(a)

    def quant(self,name,expert=None,round_weight=True):
        q=json.loads(self.tensor(name+'.quant_state.bitsandbytes__nf4').tobytes())
        shape=q['shape']; size=int(np.prod(shape[-2:])); offset=0 if expert is None else expert*size
        assert len(shape)==2 if expert is None else len(shape)==3
        p=self.tensor(name).reshape(-1)[offset//2:(offset+size)//2]
        codes=np.empty(size,dtype=np.uint8);codes[::2]=p>>4;codes[1::2]=p&15
        start=offset//q['blocksize']; end=(offset+size)//q['blocksize']
        scale_codes=self.tensor(name+'.absmax')[start:end]
        if 'nested_offset' in q:
            nested=self.tensor(name+'.nested_absmax')
            scale=self.tensor(name+'.nested_quant_map')[scale_codes]*nested[np.arange(start,end)//q['nested_blocksize']]+np.float32(q['nested_offset'])
        else:
            scale=scale_codes
        w=self.tensor(name+'.quant_map')[codes].reshape(-1,q['blocksize'])*scale[:,None]
        w=w.reshape(shape[-2:])
        return bf(w) if round_weight else w

    def linear(self,name,x):
        return bf(np.asarray(x)@self.quant(name).T)

    def head(self,x):
        w=self.tensor('lm_head.weight'); y=np.empty(w.shape[0],np.float32)
        for start in range(0,len(y),2048):
            block=(w[start:start+2048].astype(np.uint32)<<16).view(np.float32)
            y[start:start+len(block)]=bf(block@x)
        return y


def norm(x,w,axis=-1):
    return bf(x/np.sqrt(np.mean(x*x,axis=axis,keepdims=True)+1e-6)*(1+w))


def hc(c,h,prefix,inject=True):
    n=norm(h.reshape(4,2560),c.dense(prefix+'hc_norm.weight').reshape(4,2560)).reshape(-1)
    down=bf(c.dense(prefix+'input_mix_weight_down.weight')@n)/4
    act=bf(down*sigmoid(down))
    mix=bf(sigmoid(bf(c.dense(prefix+'input_mix_weight_up.weight')@act)))
    x=bf(np.sum((mix*n).reshape(4,2560),axis=0)/4)
    injection=2*sigmoid(bf(c.dense(prefix+'block_inject_weight.weight')@n)/4) if inject else None
    return x,injection


def rope(x,position):
    # Text-only partial RoPE, 64 of 256 dimensions, split-half rotation.
    angle=np.float32(position)/np.power(np.float32(1e7),np.arange(32,dtype=np.float32)/32)
    c,s=np.cos(angle),np.sin(angle);out=x.copy()
    out[:,:32]=bf(x[:,:32]*c-x[:,32:64]*s)
    out[:,32:64]=bf(x[:,32:64]*c+x[:,:32]*s)
    return out


def step(c,hidden,token,position,keys,values,fixed_route=None):
    prefix='mtp.layers.0.'
    ew=c.tensor('model.language_model.embed_tokens.weight')[token]
    embedding=(ew.astype(np.uint32)<<16).view(np.float32)
    e=c.linear('mtp.fc_embedding.weight',norm(embedding,c.dense('mtp.pre_fc_norm_embedding.weight')))
    h=c.linear('mtp.fc_hidden.weight',norm(hidden,c.dense('mtp.pre_fc_norm_hidden.weight')).reshape(4,2560))
    h=bf(h+e).reshape(-1)
    stages={'fc':h.copy()}
    x,inj=hc(c,h,prefix+'attn_hyper_connection.')
    stages['attn_x']=x.copy()
    qg=c.linear(prefix+'self_attn.q_proj.weight',x).reshape(24,512)
    k=c.linear(prefix+'self_attn.k_proj.weight',x).reshape(2,256)
    v=c.linear(prefix+'self_attn.v_proj.weight',x).reshape(2,256)
    q=bf(qg[:,:256]/np.sqrt(np.mean(qg[:,:256]**2,axis=1,keepdims=True)+1e-6)*bf(1+c.dense(prefix+'self_attn.q_norm.weight')))
    k=bf(k/np.sqrt(np.mean(k*k,axis=1,keepdims=True)+1e-6)*bf(1+c.dense(prefix+'self_attn.k_norm.weight')))
    q,k=rope(q,position),rope(k,position)
    stages.update(q=q.reshape(-1),k=k.reshape(-1),v=v.reshape(-1),gate=qg[:,256:].reshape(-1))
    keys.append(k);values.append(v)
    kk=np.stack(keys);vv=np.stack(values);attn=[]
    for head in range(24):
        scores=kk[:,head//12]@q[head]/16
        probs=np.exp(scores-scores.max());probs/=probs.sum()
        # Decode attention materializes BF16 before its fused gate multiply.
        attn.append(bf(bf(probs@vv[:,head//12])*sigmoid(qg[head,256:])))
    stages['attn_out']=np.asarray(attn).reshape(-1)
    y=c.linear(prefix+'self_attn.o_proj.weight',stages['attn_out'])
    h=bf(h.reshape(4,2560)+inj[:,None]*y).reshape(-1)
    x,inj=hc(c,h,prefix+'mlp_hyper_connection.')
    stages['moe_x']=x.copy()
    route=bf(c.dense(prefix+'mlp.gate.weight')@x)
    stages['router']=route.copy()
    selected=np.argsort(-route,kind='stable')[:10] if fixed_route is None else np.array(fixed_route)
    rw=np.exp(route[selected]-route.max());rw/=rw.sum()
    g=c.quant(prefix+'mlp.shared_expert.gate_proj.weight',round_weight=False)@x
    u=c.quant(prefix+'mlp.shared_expert.up_proj.weight',round_weight=False)@x
    shared=c.linear(prefix+'mlp.shared_expert.down_proj.weight',bf(g*sigmoid(g)*u))
    shared_gate=bf(c.dense(prefix+'mlp.shared_expert_gate.weight')@x)
    total=np.zeros(2560,np.float32)
    for expert,weight in zip(selected,rw):
        gu=c.quant(prefix+'mlp.experts.gate_up_proj',int(expert),False)@x
        activation=bf(gu[:640]*sigmoid(gu[:640])*gu[640:])
        total+=weight*(c.quant(prefix+'mlp.experts.down_proj',int(expert),False)@activation)
    y=bf(bf(total)+sigmoid(shared_gate)*shared)
    h=bf(h.reshape(4,2560)+inj[:,None]*y).reshape(-1)
    mixed,_=hc(c,h,'mtp.hyper_connection_mixer.',False)
    return h,mixed,c.head(mixed),selected.tolist(),stages


def compare(reference,actual):
    assert np.isfinite(reference).all() and np.isfinite(actual).all()
    error=np.asarray(actual,np.float64)-reference
    return dict(nrmse=float(np.linalg.norm(error)/max(np.linalg.norm(reference.astype(np.float64)),1e-12)),max_abs=float(np.abs(error).max()))


def run(directory,prefix):
    c=Checkpoint(directory);keys=[];values=[];results=[];gpu_keys=[];gpu_values=[]
    refs=[json.loads(s.split('[MTP REF] ',1)[1]) for s in Path(prefix+'.log').read_text(encoding='utf-8').splitlines() if s.startswith('[MTP REF] ')]
    assert [r['position'] for r in refs]==list(range(4))
    for r in refs:
        p=r['position'];base=prefix+'.ref'+str(p)
        oldk,oldv=list(keys),list(values)
        h,m,logits,route,stages=step(c,read_bf(base+'.input.bf16'),r['token'],p,keys,values)
        gpu=[read_bf(base+'.'+name+'.bf16') for name in ['hidden','mixed','logits']]
        metrics={name:compare(a,b) for name,a,b in zip(['hidden','mixed','logits'],[h,m,logits],gpu)}
        a=logits.astype(np.float64);b=gpu[2].astype(np.float64)
        a-=a.max();a-=np.log(np.exp(a).sum());b-=b.max();b-=np.log(np.exp(b).sum())
        metrics['kl']=float(np.sum(np.exp(a)*(a-b)))
        metrics['route_matches']=route==r['experts'];metrics['top1_matches']=int(logits.argmax())==int(gpu[2].argmax())
        metrics['position']=p;metrics['passed']=all(metrics[n]['nrmse']<=0.03 for n in ['hidden','mixed','logits']) and metrics['kl']<=0.01
        metrics['cpu_route']=route;metrics['gpu_route']=r['experts']
        metrics['stages']={name:compare(value,read_bf(base+'.'+name+'.bf16')) for name,value in stages.items() if Path(base+'.'+name+'.bf16').exists()}
        if Path(base+'.router.bf16').exists():
            gpu_scores=read_bf(base+'.router.bf16');s=np.sort(stages['router'])[::-1];g=np.sort(gpu_scores)[::-1]
            metrics['rank10_gap']={'cpu':float(s[9]-s[10]),'gpu':float(g[9]-g[10])}
        if Path(base+'.q.bf16').exists():
            q=read_bf(base+'.q.bf16').reshape(24,256);gate=read_bf(base+'.gate.bf16').reshape(24,256)
            gpu_keys.append(read_bf(base+'.k.bf16').reshape(2,256));gpu_values.append(read_bf(base+'.v.bf16').reshape(2,256))
            kk=np.stack(gpu_keys);vv=np.stack(gpu_values);attention=[]
            for head in range(24):
                scores=kk[:,head//12]@q[head]/16;probs=np.exp(scores-scores.max());probs/=probs.sum()
                attention.append(bf(bf(probs@vv[:,head//12])*sigmoid(gate[head])))
            metrics['attention_with_gpu_inputs']=compare(np.asarray(attention).reshape(-1),read_bf(base+'.attn_out.bf16'))
        if route!=r['experts']:
            ch,cm,cl,_,_=step(c,read_bf(base+'.input.bf16'),r['token'],p,oldk,oldv,r['experts'])
            metrics['conditioned_on_gpu_route']={name:compare(a,b) for name,a,b in zip(['hidden','mixed','logits'],[ch,cm,cl],gpu)}
        results.append(metrics);print(json.dumps(metrics),flush=True)
    return dict(reference=__doc__,contract={'nrmse':0.03,'kl':0.01,'top1':'diagnostic only'},results=results,passed=all(r['passed'] for r in results),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('prefix');ap.add_argument('--model',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();result=run(a.model,a.prefix);Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    raise SystemExit(0 if result['passed'] else 1)
