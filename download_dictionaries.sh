#!/bin/sh
set -e

SRC=en
TGT=(
    "ar"
    "bg"
    "de"
    "el"
    "es"
    "fr"
    "hi"
    "ru"
    "th"
    "vi"
    "zh"
)


WIKI_DUMP_NAME=${SRC}wiki-20191020-pages-articles-multistream1.xml-p10p30302.bz2
WIKI_DUMP_DOWNLOAD_URL=https://dumps.wikimedia.org/${LG}wiki/latest/$WIKI_DUMP_NAME

# download latest Wikipedia dump in chosen language
echo "Downloading the latest $LG-language Wikipedia dump from $WIKI_DUMP_DOWNLOAD_URL..."
wget -c $WIKI_DUMP_DOWNLOAD_URL -P /scratch/aishikc/
echo "Succesfully downloaded the latest $LG-language Wikipedia dump to $WIKI_DUMP_NAME"