

#include <iostream>

#include "transportation_1d.hpp"

int main() {
  Transportation1dSolver pb = Transportation1dSolver::read(std::cin);
  pb.check();
  pb.run();
  auto sol = pb.computeSolution();
  pb.checkSolutionValid(sol);
  pb.checkSolutionOptimal(sol);
  pb.write(std::cout);
  Transportation1dSolver::writeSolution(sol, std::cout);
  return 0;
}