#!/usr/bin/python
#-*- coding: utf-8 -*-

GRADES = [
    "6A+",
    "6B",
    "6B+",
    "6C",
    "6C+",
    "7A",
    "7A+",
    "7B",
    "7B+",
    "7C",
    "7C+",
    "8A",
    "8A+",
    "8B",
    "8B+",
]

def get_grade(boulder):
    grade = 0  # Default grade
    return GRADES[grade]

if __name__ == "__main__":
    boulder = [(0, 0), (1, 0), (2, 0)]  # Example boulder holds
    grade = get_grade(boulder)
    print(f"Boulder grade: {grade}")