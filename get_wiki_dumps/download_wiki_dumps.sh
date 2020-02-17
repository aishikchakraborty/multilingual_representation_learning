#!/bin/sh
set -e

LG=$1
# WIKI_DUMP_NAME=${LG}wiki-latest-pages-articles.xml.bz2
# https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream1.xml-p10p30302.bz2
WIKI_DUMP_NAME=${LG}wiki-latest-pages-articles-multistream.xml.bz2
WIKI_DUMP_DOWNLOAD_URL=https://dumps.wikimedia.org/${LG}wiki/latest/$WIKI_DUMP_NAME

# download latest Wikipedia dump in chosen language
echo "Downloading the latest $LG-language Wikipedia dump from $WIKI_DUMP_DOWNLOAD_URL..."
wget -c $WIKI_DUMP_DOWNLOAD_URL -P /scratch/aishikc/
echo "Succesfully downloaded the latest $LG-language Wikipedia dump to $WIKI_DUMP_NAME"