from clustering_scripts import john_crystal_feedback, john_crystal_actions, john_crystal_progression
from clustering_scripts import john_lakeland_actions, john_lakeland_feedback, john_lakeland_progression

def main():
    funcs = [john_crystal_feedback.main, john_crystal_progression.main, john_crystal_actions.main]
    for f in funcs:
        f()


if __name__ == '__main__':
    main()