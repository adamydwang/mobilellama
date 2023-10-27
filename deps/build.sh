install_path=`pwd`/..
cd sentencepiece
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${install_path} ..
make
make install

