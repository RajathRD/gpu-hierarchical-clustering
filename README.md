# gpu-project

First, do  `chmod -R +x *.sh` and `./cpu_compile.sh` to compile.

Then, to run for random dataset of size N x M, cd to root and do:

`./cpu_clustering N M`

To run unit tests, cd to root, do:
 
 `./test.sh`

 If all the tests passed, at the bottom of the output, you will see:

 `test1.txt`

 `test2.txt`

 `test3.txt`
 
 `...`

 If a test failed, there will be diff output below it.