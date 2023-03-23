

#include <iostream>

#include "transportation_1d.hpp"

int main() {
  Transportation1d pb = Transportation1d::read(std::cin);
  pb.check();
  auto sol = pb.solve();
  pb.checkSolutionValid(sol);
  pb.checkSolutionOptimal(sol);
  pb.write(std::cout);
  Transportation1d::writeSolution(sol, std::cout);
  return 0;
}