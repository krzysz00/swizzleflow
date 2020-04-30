#!/usr/bin/env zsh
echo "Level 3"
for i in {0..2}
do
    racket $1 | grep 'time'
done

sed -i -e 's/\?sw-xform/?sw-xform-easy/g' -e 's/\?cond/\?cond-easy/g' $1

echo "Level 1"
for i in {0..2}
do
    racket $1 | grep 'time'
done

sed -i -e 's/\?sw-xform-easy/?sw-xform/g' -e 's/\?cond-easy/?cond/g' $1
