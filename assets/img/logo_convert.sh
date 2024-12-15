#!/bin/bash

 convert all FLkit_logo*.svg to png
sips -s format png FLkit_logo.svg --out  FLkit_logo.png --resampleWidth 2760 --resampleHeight 1024
sips -s format png FLkit_logo_inverted.svg --out  FLkit_logo_inverted.png --resampleWidth 2760 --resampleHeight 1024
sips -s format png FLkit_logo_condensed.svg --out  FLkit_logo_condensed.png --resampleWidth 1024 --resampleHeight 1024
sips -s format png FLkit_logo_condensed_inverted.svg --out  FLkit_logo_condensed_inverted.png --resampleWidth 1024 --resampleHeight 1024

# Convert svg to png with sips
sips -s format png FLkit_logo_condensed.svg --out favicon-16x16.png --resampleWidth 16 --resampleHeight 16
sips -s format png FLkit_logo_condensed.svg --out favicon-32x32.png --resampleWidth 32 --resampleHeight 32

# Convert svg to ico with sips
sips -s format ico FLkit_logo_condensed.svg --out favicon.ico  --resampleWidth 32 --resampleHeight 32

sips -s format png FLkit_logo_condensed.svg --out apple-touch-icon.png --resampleWidth 192 --resampleHeight 192
sips -s format png FLkit_logo_condensed.svg --out android-chrome-192x192.png --resampleWidth 192 --resampleHeight 192
sips -s format png FLkit_logo_condensed.svg --out android-chrome-512x512.png --resampleWidth 512 --resampleHeight 512


