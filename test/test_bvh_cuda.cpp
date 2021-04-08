#include <pch.h>
#include <Loader.hpp>
#include <Scene.h>
#include <iostream>
#include <Render.h>

int main() {
    Loader<Actor> loaderActor;
    Actor* actor = new Actor();
    loaderActor.load(actor, "D:/resources/cornellbox/floor.obj");
    loaderActor.load(actor, "D:/resources/cornellbox/left.obj");
    loaderActor.load(actor, "D:/resources/cornellbox/right.obj");
    loaderActor.load(actor, "D:/resources/cornellbox/light.obj");
    loaderActor.load(actor, "D:/resources/cornellbox/shortbox.obj");
    loaderActor.load(actor, "D:/resources/cornellbox/tallbox.obj");
    Scene scene;
    scene.Actors.push_back(actor);
    scene.setupDevice();
    scene.computeBoundingBoxes();
    scene.buildBVH();

    Render render;
    render.rend(&scene, 1024, 1024, 16);
    // render.rend(&scene, 32, 32, 16);
    render.output();

    printf("kernel:\n");
    cudaDeviceSynchronize();
    
    return 1;
}