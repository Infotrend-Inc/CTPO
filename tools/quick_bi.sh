#!/bin/bash

td=$1
if [ ! -d $td ]; then
  echo "Usage: $0 <target_dir>"
  exit 1
fi

of=$td/BuildInfo.txt
echo "" > $of
if [ ! -f $of ]; then
  echo "Unable to create $of, aborting"
  exit 1
fi

# First part of the BuildInfo is from the env.txt file

if [ ! -f $td/env.txt ]; then
    echo "Unable to find $td/env.txt, aborting"
    exit 1
fi

cat $td/env.txt >> $of

# Second part is in the "System--" file within $td the directory

py=$(find $td -name "System--*" -type f)
if [ ! -f $py ]; then
    echo "Unable to find a \"System--*\" file in $td, aborting"
    exit 1
fi

# extract the part after the [system details] line
perl -ne 'print if /\[system/ .. eof' $py | perl -ne 'next if (m%^\s*$%); next if (m%\[%); s%\:\s*%=%;print' >> $of

exit 0
