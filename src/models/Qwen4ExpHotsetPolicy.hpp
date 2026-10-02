#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// Strata's per-layer frequency/hysteresis policy. Compute placement never feeds
// this policy: every actual router contribution is observed, including CPU misses.
struct Qwen4ExpHotsetPolicy {
    struct Swap { int incoming, outgoing; float gain; };
    int layers = 0, experts = 0;
    std::vector<float> usage;
    std::vector<uint64_t> decode, prefill, unique;
    std::vector<int> ranking;
    void Init(int l, int e) {
        layers=l; experts=e; const int n=l*e;
        usage.assign(n,0); decode.assign(n,0); prefill.assign(n,0); unique.assign(n,0);
        ranking.resize(n); std::iota(ranking.begin(),ranking.end(),0);
    }
    void Observe(int layer, const int* ids, int rows, int topk) {
        std::array<bool,512> seen{};
        if(experts>512)throw std::invalid_argument("Hotset expert dimension");
        for(int i=0;i<rows*topk;++i) {
            if(ids[i]<0 || ids[i]>=experts)throw std::invalid_argument("Invalid hotset route");
            const int k=layer*experts+ids[i];
            if(rows==1) {++decode[k]; usage[k]+=1;} else ++prefill[k];
            if(!seen[ids[i]]) {++unique[k]; seen[ids[i]]=true;}
        }
    }
    std::vector<Swap> Select(const std::vector<int>& resident, int limit) const {
        std::vector<Swap> result;
        for(int l=0;l<layers;++l) {
            std::vector<int> candidates,victims;
            for(int e=0;e<experts;++e) {const int k=l*experts+e;
                if(resident[k]>=0)victims.push_back(k);
                else if(usage[k]>=2.0f)candidates.push_back(k);
            }
            auto hotter=[&](int a,int b){return usage[a]!=usage[b]?usage[a]>usage[b]:a<b;};
            auto colder=[&](int a,int b){return usage[a]!=usage[b]?usage[a]<usage[b]:a<b;};
            std::sort(candidates.begin(),candidates.end(),hotter);
            std::sort(victims.begin(),victims.end(),colder);
            for(size_t i=0;i<std::min(candidates.size(),victims.size());++i) {
                const float gain=usage[candidates[i]]-usage[victims[i]];
                if(gain<1.5f)break;
                result.push_back({candidates[i],victims[i],gain});
            }
        }
        std::sort(result.begin(),result.end(),[](const Swap& a,const Swap& b){
            return a.gain!=b.gain?a.gain>b.gain:a.incoming<b.incoming;});
        if(result.size()>size_t(limit))result.resize(limit);
        return result;
    }
    void Decay() {for(float& v:usage)v*=0.7f;}
    void Save(const std::string& path,const std::string& identity) const {
        std::ofstream out(path);if(!out)throw std::runtime_error("Cannot write hotset profile");
        std::vector<int> order=ranking;
        std::sort(order.begin(),order.end(),[&](int a,int b){
            if(decode[a]!=decode[b])return decode[a]>decode[b];
            return prefill[a]!=prefill[b]?prefill[a]>prefill[b]:a<b;});
        out<<"AILA_Q4E_HOTSET 1\n"<<identity<<'\n'<<layers<<' '<<experts<<' '<<order.size()<<'\n';
        for(int k:order)out<<k<<' '<<decode[k]<<' '<<prefill[k]<<' '<<unique[k]<<'\n';
        if(!out)throw std::runtime_error("Incomplete hotset profile write");
    }
    void Load(const std::string& path,const std::string& identity) {
        std::ifstream in(path);std::string magic,id;int version=0,l=0,e=0,n=0;
        if(!(in>>magic>>version>>id>>l>>e>>n) || magic!="AILA_Q4E_HOTSET" || version!=1 ||
            id!=identity || l!=layers || e!=experts || n!=layers*experts)
            throw std::runtime_error("Hotset profile identity or dimensions mismatch");
        std::vector<int> order;std::vector<bool> seen(n,false);
        for(int i=0;i<n;++i) {int key;uint64_t d,p,u;
            if(!(in>>key>>d>>p>>u) || key<0 || key>=n || seen[key])
                throw std::runtime_error("Invalid/duplicate hotset profile key");
            seen[key]=true;order.push_back(key);
        }
        std::string trailing;if(in>>trailing)throw std::runtime_error("Trailing hotset profile data");
        ranking=std::move(order);
        // Offline counts rank the initial tier; online usage starts empty (Strata).
    }
};
