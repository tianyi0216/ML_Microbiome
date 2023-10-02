# go through all direcotries and unzip all files

for dir in $(ls -d */); do
    echo $dir
    cd $dir
    for file in $(ls *.zip); do
        echo $file
        unzip $file
    done
    cd ..
done