# KGBench datatype

KGBench datasets contain bty-to-string encoded literals. These string literals encode byte-level data, potentially containing images, video or audio (although only images are currently supplied).

 They have the following format

```http://krrvu.github.io/kgbench/dt#[media]-[string-encoding][-media-encoding]```

Media types can be one of: `image`, `audio`, `video`. `string-encoding` is `base64` for a base64 encoding. Other encodings are currently not specified.

The media-encoding may be ommitted if it can be inferred from the byte-level representation of the file. If it is present 
it serves as a _hint_ towards the way the byte-level representation should be decoded, in the same way that file extensions do

Some examples:
```
# for an image without specified codec
http://krrvu.github.io/kgbench/dt#image-base64

# for a jpg-encoded image
http://krrvu.github.io/kgbench/dt#image-base64-jpg

# for a wav-encoded audio file
http://krrvu.github.io/kgbench/dt#audio-base64-wav

# for a video file we specify either no internal encoding or the container type

http://krrvu.github.io/kgbench/dt#video-base64
http://krrvu.github.io/kgbench/dt#video-base64-mkv
```
