#include "finders.h"
#include "util.h"

#include <stdio.h>
#include <time.h>
#include <stdbool.h>

bool containsBirch(int* biomeIds, int sx, int sz) {
  bool birch = false;
  for (int x = 0; x < sx; x++) {
    for (int z = 0; z < sz; z++){
      int biome = biomeIds[x + z*sx];
      if (biome == birchForest || biome == birchForestHills) birch = true;
    }
  }
  return birch;
}


bool isOcean(int biome) {
  return (biome == ocean) || (biome == warm_ocean) || (biome == cold_ocean)
      || (biome == lukewarm_ocean) || (biome == deep_ocean);
}

bool isDarkForest(int biome) {
  return (biome == dark_forest) || (biome == dark_forest_hills);
}


int mod (int a, int b) {
  return ((b + (a%b)) % b);

}

int main(int argc, char** argv) {

  Generator g;
  setupGenerator(&g, MC_1_19, 0);

  uint64_t seed = 64149200LL;
  applySeed(&g, DIM_OVERWORLD, seed);


  clock_t start, stop;
  start = clock();

  // Check withing a 100-block radius.
  Range r;
  r.scale = 16; // 1 Chunk at a time
  r.sx = 1, r.sz = 3;
  r.y = 40, r.sy = 1;
  int* biomeIds = allocCache(&g, r);

  Pos sPos;

  // Defines # of regions to scan. ~23440 covers the entire world.
  int regions = 23440;

  // Matchs one, two, or three checks.
  int viable = 0;
  int doubleViable = 0;
  int tripleViable = 0;
  for (int x = -regions; x < regions; x++) {
    for (int z = -regions; z < regions; z++) {
      int res = getStructurePos(Mansion, MC_1_19, seed, x, z, &sPos);
      if (res == 0) {
        printf("[ERROR] Could not get structure pos\n");
        continue;
      }
      if (x == -regions && (z == -regions)) fprintf(stderr, "Range of search: %d %d\n", sPos.x, sPos.z);

      if ( (mod(x, 2000) == 0) && (z == 0)) {
        float time = (clock() - start) / (float)(CLOCKS_PER_SEC);
        fprintf(stderr, "[PROGRESS] x_region is %d in %f seconds\n", x, time);
      }

      // Enforce Cloud constraint
      int modZ = mod(sPos.z, 6000);
      if (modZ < 5410 || modZ > 5580) {
        continue;
      }

      // Make sure a mansion can actually spawn here.
      if (!isViableStructurePos(Mansion, &g, sPos.x, sPos.z, 0)) {
        continue;
      }

      // We see at 7:04/7:05 of the video that the mansion entrance is facing the
      // +x direction, which corresponds to  rot=0
      uint64_t rng = chunkGenerateRnd(seed, sPos.x >> 4, sPos.z >> 4);
      int rot = nextInt(&rng, 4);
      if (rot != 0) continue;


      int biomeZ = getBiomeAt(&g, 16, sPos.x/16 - 1,  40, sPos.z/16 - 8);
      if (!isOcean(biomeZ)) {
        continue;
      }

      // We found a mansion with correct orientation and plausible Z position.
      viable++;

      r.x = sPos.x / (r.scale) + 1;
      r.z = sPos.z / (r.scale) + 1;
      genBiomes(&g, biomeIds, r);
      if (containsBirch(biomeIds, r.sx, r.sz)) {
        doubleViable++;

        // From 7:05 on video, we know there should be some sort of ocean chunks
        // in the -z from mansion and -x from mansion.

        // These [getbiomeAt] functions that chunk coordinates.
        int biomeX = getBiomeAt(&g, 16, sPos.x/16 - 11, 40, sPos.z/16 - 1);
        int biomeXZ = getBiomeAt(&g, 16, sPos.x/16 - 8, 40, sPos.z/16 - 5);

        int fBiome  = getBiomeAt(&g, 16, sPos.x/16 + 1, 40, sPos.z/16);
        int fBiome2 = getBiomeAt(&g, 16, sPos.x/16 + 2, 40, sPos.z/16);
        int fBiome3 = getBiomeAt(&g, 16, sPos.x/16 + 3, 40, sPos.z/16);

        if (isOcean(biomeZ) && isOcean(biomeX) && isDarkForest(fBiome) && isDarkForest(fBiome2)
            && isOcean(biomeXZ) && isDarkForest(fBiome3)
            ) {
          tripleViable++;
            // uncomment following line to print best guesses.
            //printf("/tp %d 130 %d\n", sPos.x, sPos.z);
        }
      }
    }
  }

  stop = clock();

  float percent = viable / (float)(regions * regions);
  float time = (stop - start) / (float)(CLOCKS_PER_SEC);

  printf("Found %d mansions, or %f percent of regions\n", viable, percent);
  printf("Found %d mansions with good regions\n", doubleViable);
  printf("Found %d mansions with best region\n", tripleViable);
  printf("Runtime: %f seconds.\n", time);

  free(biomeIds);
}
