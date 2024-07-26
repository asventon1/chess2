#! /bin/sh

kill -9 $(pgrep -f 'python watch.py')
kill -9 $(pgrep -f 'python ../main.py')
