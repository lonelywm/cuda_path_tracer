#include <pch.h>
#include <Loader.hpp>
#include <iostream>


__global__
void test(Vec3f vec, float a) {
    printf("%f, %f, %f, %f", vec.x, vec.y, vec.z, a);
}

int main(int args, char* argv[]) {
    std::string path = argv[0];

    Loader<Actor> loader;
    Actor actor;

    loader.load(&actor, "D:/resources/cornellbox/floor.obj");
    loader.load(&actor, "D:/resources/cornellbox/left.obj");
    loader.load(&actor, "D:/resources/cornellbox/right.obj");
    loader.load(&actor, "D:/resources/cornellbox/light.obj");
    loader.load(&actor, "D:/resources/cornellbox/shortbox.obj");
    loader.load(&actor, "D:/resources/cornellbox/tallbox.obj");

    // std::cout << argv[0] << std::endl;
    // std::cout << actor.Meshes.size();

    Vec3f max;
    max.x = 100;
    max.y = -100;
    max.z = 1000;

    test<<<1, 1>>>(max, 999.0);

    return 1;
};