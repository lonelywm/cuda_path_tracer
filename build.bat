cd build
@REM cmake ..
cmake --build .
cd ..

copy bin\Debug\test_bvh_cuda.exe Q:\MPathTracing\bin\Debug /Y 

