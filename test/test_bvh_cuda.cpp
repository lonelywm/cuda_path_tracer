#include <pch.h>
#include <Loader.hpp>
#include <Scene.h>
#include <iostream>
#include <Render.h>

int main() {
    Loader<Actor> loaderActor;
    Actor* actor = new Actor();
    Material matLight;
    Material matRed;
    Material matGreen;
    Material matWhite;

    matRed.kd = Vec3f(0.63f, 0.065f, 0.05f);
    matGreen.kd = Vec3f(0.14f, 0.45f, 0.091f);
    matWhite.kd = Vec3f(0.725f, 0.71f, 0.68f);
    matLight.ke = (8.0f * Vec3f(0.747f + 0.058f, 0.747f + 0.258f, 0.747f) + 15.6f * Vec3f(0.740f + 0.287f, 0.740f + 0.160f, 0.740f) + 18.4f * Vec3f(0.737f + 0.642f, 0.737f + 0.159f, 0.737f));

    loaderActor.load(actor, "d:/resources/cornellbox/light.obj");
    loaderActor.load(actor, "d:/resources/cornellbox/floor.obj");
    loaderActor.load(actor, "d:/resources/cornellbox/left.obj");
    loaderActor.load(actor, "d:/resources/cornellbox/right.obj");
    loaderActor.load(actor, "d:/resources/cornellbox/shortbox.obj");
    loaderActor.load(actor, "d:/resources/cornellbox/tallbox.obj");
    loaderActor.load(actor, "d:/resources/cornellbox/face.obj");

    actor->Meshes[0]->Mtrl = matLight;
    actor->Meshes[1]->Mtrl = matWhite;
    actor->Meshes[2]->Mtrl = matRed;
    actor->Meshes[3]->Mtrl = matGreen;
    actor->Meshes[4]->Mtrl = matWhite;
    actor->Meshes[5]->Mtrl = matWhite;
    actor->Meshes[6]->Mtrl = matWhite;

    actor->Meshes[6]->transfer(Vec3f(278, 273, 300));
    actor->Meshes[0]->scale(Vec3f(2.4));

    Scene scene;
    scene.Actors.push_back(actor);
    scene.setupDevice();
    scene.computeBoundingBoxes();
    scene.buildBVH();

    Render render;

    render.init(&scene, 1024, 1024, 1, 6);
    // render.init(&scene, 128, 128, 64, 6);
    // render.init(&scene, 32, 32, 16);

    render.phong();
    
    render.output();

    printf("kernel:\n");
    
    return 1;
}