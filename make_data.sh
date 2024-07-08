#!/bin/bash
echo '[' > $2
head -n $3 $1 | sed '$!s/$/,/' >> $2
echo ']' >> $2
