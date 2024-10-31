#!/bin/bash
preproc_dir="../data/preproc"
if [[ -d $preproc_dir ]];
then
    rm -rf $preproc_dir
fi
mkdir $preproc_dir