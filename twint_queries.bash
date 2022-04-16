#!/bin/bash

: '
Additional options
--verified
'

date=$(date +%m%d%y)
directory="data/${date}"
mkdir -p ${directory}
for subject in "russia" "ukraine" "putin" "zelensky" ; do
    output_file="${directory}/${subject}.twint"
    twint -s "#${subject}" -o ${output_file} --lang "en" --min-likes 42
    grep -Pv "[\x80-\xFF]" ${output_file} > _tmp
    mv _tmp ${output_file}
done

grep -vi ukraine "${directory}/russia.twint" | grep -vi ukran | grep -vi ukrain > _tmp
mv _tmp "${directory}/russia.twint"

grep -vi russia "${directory}/ukraine.twint" > _tmp
mv _tmp "${directory}/ukraine.twint"

grep -v zelensky "${directory}/putin.twint" > _tmp
mv _tmp "${directory}/putin.twint"

grep -v putin "${directory}/zelensky.twint" > _tmp
mv _tmp "${directory}/zelensky.twint"

