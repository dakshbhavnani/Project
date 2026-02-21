import pandas as pd

def get_career_info(career_df, career):
    row = career_df[career_df.iloc[:,0] == career]
    if len(row) == 0:
        return None
    return row.iloc[0]

def skill_gap_analysis(student_skills, required_skills):
    student = set([s.strip().lower() for s in student_skills])
    required = set([s.strip().lower() for s in required_skills.split("|")])

    matched = student.intersection(required)
    missing = required.difference(student)

    return list(matched), list(missing)