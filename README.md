# Motion Matching & Code vs Data Driven Displacement

This repo contains the source code for all the demos from [this article](https://theorangeduck.com/page/code-vs-data-driven-displacement).

It also contains basic example implementations of Motion Matching and Learned Motion Matching in the style of [this paper](https://theorangeduck.com/page/learned-motion-matching).

# Installation

This demo uses [raylib](https://www.raylib.com/) and [raygui](https://github.com/raysan5/raygui) so you will need to first install those. Once installed, the demo itself is a pretty straight forward to make - just compile `controller.cpp`.

I've included a basic `Makefile` which you can use if you are using raylib on Windows. You may need to edit the paths in the `Makefile` but assuming default installation locations you can just run `Make`.

If you are on Linux or another platform you will probably have to hack this `Makefile` a bit.

# Web Demo

If you want to compile the web demo you will need to first [install emscripten](https://github.com/raysan5/raylib/wiki/Working-for-Web-%28HTML5%29). Then you should be able to (on Windows) run `emsdk_env` followed by `make PLATFORM=PLATFORM_WEB`. You then need to run `wasm-server.py`, and from there will be able to access `localhost:8080/controller.html` in your web browser which should contain the demo.

# Learned Motion Matching

Most of the code and logic you can find in `controller.cpp`, with the Motion Matching search itself in `database.h`. The structure of the code is very similar to the previously mentioned [paper](https://theorangeduck.com/media/uploads/other_stuff/Learned_Motion_Matching.pdf) but not identical in all respects. For example, it does not contain some of the briefly mentioned optimizations to the animation database storage and there are no tags used to disambiguate walking and running.

If you want to re-train the networks you need to look in the `resources` folder. First you will need to run `train_decompressor.py`. This will use `database.bin` and `features.bin` to produce `decompressor.bin`, which represents the trained decompressor network, and `latent.bin`, which represents the additional features learned for each frame in the database. It will dump also out some images and `.bvh` files you can use to examine the progress (as well as write Tensorboard logs to the `resources/runs` directory). Once the decompressor is trained and you have a well trained network and corresponding `latent.bin`, you can then train the stepper and the projector (at the same time) using `train_stepper.py` and `train_projector.py`. Both of these will also output networks (`stepper.bin` and `projector.bin`) as well as some images you can use to get a rough sense of the progress and accuracy.

The data required if you want to regenerate the animation database is from [this dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) which is licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License (unlike the code, which is licensed under MIT).

If you re-generate the database you will also need to re-generate the matching database `features.bin`, which is done every time you re-run the demo. Similarly if you change the weights or any other properties that affect the matching the database will need to be re-generated and the networks re-trained.
