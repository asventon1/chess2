#! /bin/sh
for value in {0..40}
do
  echo '[' > 'data/data'$value'.json'
  sed -n $((value*1000000+1))','$(((value+1)*1000000))'p;'$(((value+1)*1000000))'q' $1 | sed '$!s/$/,/' >> 'data/data'$value'.json'
  echo ']' >> 'data/data'$value'.json'
done
