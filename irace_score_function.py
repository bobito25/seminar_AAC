# https://github.com/SergioSJS/python-irace/tree/master

import argparse
import logging
import sys

from solve_tsp import solve_tsp_from_file

def main(POP, TNRMT, MUTPB, CXPB, DATFILE):
	# just a test
	#score = MUTPB*POP/100
	#score = float(score)
	#score = score - float(CXPB)
	
	#score = float((POP-1)**2 + (CXPB-2)**2 + (MUTPB-3)**2 + 1)
	#if score < 0:
	#	score = 0
	score = solve_tsp_from_file(params={"population_size": POP, "tournament_selection_size": TNRMT, "mutation_rate": MUTPB, "crossover_rate": CXPB, "target": 1.0}, filename="../genetic_tsp/TSP.txt")

	# save the fo values in DATFILE
	with open(DATFILE, 'w') as f:
		f.write(str(score))

if __name__ == "__main__":
	# just check if args are ok
	with open('args.txt', 'w') as f:
		f.write(str(sys.argv))
	
	# loading example arguments
	ap = argparse.ArgumentParser(description='Feature Selection using GA with DecisionTreeClassifier')
	ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
	# 3 args to test values
	ap.add_argument('--pop', dest='pop', type=int, required=True, help='Population size')
	ap.add_argument('--tnrmt', dest='tnrmt', type=int, required=True, help='Tournament selection size')
	ap.add_argument('--mut', dest='mut', type=float, required=True, help='Mutation probability')
	ap.add_argument('--cros', dest='cros', type=float, required=True, help='Crossover probability')
	# 1 arg file name to save and load fo value
	ap.add_argument('--datfile', dest='datfile', type=str, required=True, help='File where it will be save the score (result)')

	args = ap.parse_args()
	logging.debug(args)
	# call main function passing args
	main(args.pop, args.tnrmt, args.mut, args.cros, args.datfile)