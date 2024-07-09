#!/bin/bash
echo '[' > $2
cat $1 | sed '$!s/$/,/' >> $2
echo ']' >> $2
