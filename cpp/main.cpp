

#include <iostream>

#include "transportation_1d.hpp"

int main() {
  Transportation1d pb = Transportation1d::read(std::cin);
  Transportation1d::Solution expected =
      Transportation1d::readSolution(std::cin);
  pb.check();
  auto sol = pb.solve();
  pb.checkSolutionValid(sol);
  pb.write(std::cout);
  Transportation1d::writeSolution(sol, std::cout);
  if (pb.cost(expected) != pb.cost(sol)) {
    std::cerr << "Solution cost obtained does not match expected: "
              << pb.cost(sol) << " vs " << pb.cost(expected) << std::endl;
    exit(1);
  }
  return 0;
}