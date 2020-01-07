#!/bin/bash

echo "Start installing necessary packages"

# Check for xcode-select; install if we don't have it
# for more information, see: https://help.apple.com/xcode/mac/current/#/devc8c2a6be1
if test ! $(which xcode-select); then 
    echo "Installing xcode-select..."
    xcode-select --install
fi

# Check for Homebrew; install if we don't have it
# for more information, see: https://brew.sh/
if test ! $(which brew); then
    echo "Installing homebrew..."
    ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
fi

# Check for wget; install if we don't have it
# for more info, see: https://formulae.brew.sh/formula/wget
if test ! $(which wget); then
    echo "Installing wget..."
    brew install wget
fi

# Check for unzip; install if we don't have it
# for more info, see: https://formulae.brew.sh/formula/unzip
if test ! $(which unzip); then
    echo "Installing unzip..."
    brew install unzip
fi

# Check for anaconda; install if we don't have it
# for more info, see: https://formulae.brew.sh/cask/anaconda
if test ! $(which conda); then
    echo "Installing anaconda..."
    brew cask install anaconda
fi


echo "Finished installing necessary packages"