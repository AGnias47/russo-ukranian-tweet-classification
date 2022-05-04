files=(*twint)
for file in ${files[@]} ; do
    echo $file
    konwert utf8-ascii $file > tmp
    mv tmp $file
done
