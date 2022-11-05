
# Description

This program will locate Woodland mansions within a world using the Cubiomes library.
A sample is provided showing how we can filter based on z-position, mansion orientation,
and nearby biomes. The sample is configured to be used on the LiveOverflow "Minecraft Hacked" Youtube series world.

# Usage

Edit the core loop to filter out mansions as desired. Also change the world seed and search radius as desired.

# Building
This depends on the Cubiomes library, so first build that to produce the library file [libcubiomes.a]

Then build the mansion finder with
```
gcc find_mansion.c libcubiomes.a -lm -o find_mansions
```

and run with
```
./find_mansions
```
