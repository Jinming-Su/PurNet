function[score] = computeMAE(map, gt)
score = mean2(abs(map - gt));