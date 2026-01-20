  public int writeLength() throws IOException {
    int length = 0;
    int mapIdLength = Text.encode(mapId).limit();
    length += mapIdLength;

    length += WritableUtils.getVIntSize(mapIdLength);
    length += WritableUtils.getVIntSize(compressedLength);
    length += WritableUtils.getVIntSize(uncompressedLength);
    length += WritableUtils.getVIntSize(forReduce);

    return length;
  }