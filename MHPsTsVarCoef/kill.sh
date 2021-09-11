#!/bin/tcsh
ps aux | grep "perl" | awk '{print "kill -9 "$2}' | bash 
ps aux | grep "python" | grep "chiangwe" | awk '{print "kill -9 "$2}' | bash

