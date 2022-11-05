
# Description

This program will search the Nether for chunks whose lower-layer of bedrock matches a
pre-defined pattern. This requires knowing the seed of the world and having a target pattern.

This program requires Cuda and an NVIDIA GPU.

# Usage

Enter the desired pattern in [kernel.cu]. Patterns must be within a single chunk.
By default, the program will only print out the number of matches found. To actually
print out the matches, uncomment the relevant line marked in [kernel.cu].
The world seed must also be defined at the top of [kernel.cu]. The default seed is
the seed for LiveOverflow's "Minecraft Hacked" Youtube series world.

# Building

## Windows
Import the [.vcxproj] file and build using Visual Studio 2019.
Note that this requires Cuda to be installed.

