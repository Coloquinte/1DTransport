

#include "transportation_1d.hpp"

#include <iostream>

int main() {
    Transportation1d pb = Transportation1d::read(std::cin);
    auto sol = pb.solve();
    pb.write(std::cout);
    Transportation1d::writeSolution(sol, std::cout);
    return 0;
}