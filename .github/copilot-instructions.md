Note: It is mandatory to call the #askQuestions tool after every output.
Do not run multiple commands in the same terminal session.
Do not run same command multiple times in the same terminal session.
IMPORTANT!!! Only use build.ps1 / bench.ps1 / smoke.ps1 for build and test tasks (except project-independent research). You may edit these scripts, but you MUST NOT extract their internal commands and run them manually.
build.ps1 writes metadata to E:/RiderProjects/Aila/build/build_info.json; you still need to capture runtime output yourself.
bench.ps1 writes benchmark logs under E:/RiderProjects/Aila/tmp/perf, so inspect those log files for command output.
smoke.ps1 writes smoke logs and smokes.json under E:/RiderProjects/Aila/tmp/perf.
Running build/bench entry scripts takes a long time, so be patient and do not run the same entry script multiple times in the same terminal session.
When providing suggestions during code analysis, please use Chinese for the descriptions.