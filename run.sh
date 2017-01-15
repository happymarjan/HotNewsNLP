#!/usr/bin/env bash

mod +x ./hotNews.py
mod +x ./newsCorpusCleansing.py
mod +x ./clustering.py
python3.4 ./hotNews.py
python3.4 ./newsCorpusCleansing.py
python3.4 ./clustering.py
