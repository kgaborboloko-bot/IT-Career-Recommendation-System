#!/usr/bin/env python
# coding: utf-8

# In[2]:


# ============================================================
# IT CAREER RECOMMENDATION SYSTEM
# FILE 3: recommender.py
# ============================================================

import os

import joblib
import numpy as np

from skills_taxonomy import (
    TRACKS,
    ALL_SKILLS,
    SKILL_SPECIALISATIONS,
    recognise_skill,
    assess_input,
    taxonomy_scores,
    encode_skills,
    canonical_track
)


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = os.getcwd()

MODEL_FILE = os.path.join(
    BASE_DIR,
    "it_career_recommender_bundle.joblib"
)

# Starting weights for the hybrid recommender.
# These are heuristic weights, not validated probabilities.

ML_WEIGHT = 0.60
TAXONOMY_WEIGHT = 0.40


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

def load_recommender():

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            "Model bundle not found: " + MODEL_FILE
        )

    bundle = joblib.load(MODEL_FILE)

    required_keys = [
        "model",
        "demand_profiles",
        "junior_profiles",
        "jobs"
    ]

    for key in required_keys:

        if key not in bundle:

            raise KeyError(
                "Missing model bundle key: " + key
            )

    return bundle


# ============================================================
# 3. RECOGNISE STUDENT SKILLS
# ============================================================

def recognise_student_skills(user_input):

    results = assess_input(user_input)

    recognised = []
    unrecognised = []
    ambiguous = []

    for result in results:

        if result["status"] == "recognised":

            skill = result["skill"]

            if skill not in recognised:
                recognised.append(skill)

        elif result["status"] == "ambiguous":

            ambiguous.append(result)

        else:

            unrecognised.append(result)

    return {

        "recognised": sorted(recognised),

        "unrecognised": unrecognised,

        "ambiguous": ambiguous,

        "details": results

    }


# ============================================================
# 4. IDENTIFY SKILLS KNOWN TO ML MODEL
# ============================================================

def split_model_skills(skills, model):

    vectorizer = model.named_steps[
        "vectorizer"
    ]

    vocabulary = set(
        vectorizer.get_feature_names_out()
    )

    model_known = []
    taxonomy_only = []

    for skill in skills:

        if skill in vocabulary:

            model_known.append(skill)

        else:

            taxonomy_only.append(skill)

    return {

        "model_known": sorted(model_known),

        "taxonomy_only": sorted(taxonomy_only)

    }


# ============================================================
# 5. MACHINE LEARNING SCORES
# ============================================================

def get_ml_scores(skills, model):

    scores = {
        track: 0.0
        for track in TRACKS
    }

    if not skills:

        return scores

    encoded = encode_skills(skills)

    probabilities = model.predict_proba(
        [encoded]
    )[0]

    classes = model.classes_

    for track, probability in zip(
        classes,
        probabilities
    ):

        track = canonical_track(track)

        if track in scores:

            scores[track] += float(probability)

    return scores


# ============================================================
# 6. HYBRID SPECIALISATION SCORES
# ============================================================

def hybrid_scores(
    skills,
    model,
    ml_weight=ML_WEIGHT,
    taxonomy_weight=TAXONOMY_WEIGHT
):

    skill_groups = split_model_skills(
        skills,
        model
    )

    model_known = skill_groups[
        "model_known"
    ]

    taxonomy_only = skill_groups[
        "taxonomy_only"
    ]

    # Only use the classifier when at least one
    # skill is in its training vocabulary.

    if model_known:

        ml_scores = get_ml_scores(
            model_known,
            model
        )

    else:

        ml_scores = {
            track: 0.0
            for track in TRACKS
        }

    # The taxonomy uses every recognised skill,
    # including those unseen during ML training.

    tax_scores = taxonomy_scores(
        skills
    )

    has_ml = len(model_known) > 0

    has_taxonomy = (
        sum(tax_scores.values()) > 0
    )

    # Adjust weights depending on available evidence.

    if has_ml and has_taxonomy:

        total_weight = (
            ml_weight + taxonomy_weight
        )

        effective_ml_weight = (
            ml_weight / total_weight
        )

        effective_tax_weight = (
            taxonomy_weight / total_weight
        )

    elif has_ml:

        effective_ml_weight = 1.0
        effective_tax_weight = 0.0

    elif has_taxonomy:

        effective_ml_weight = 0.0
        effective_tax_weight = 1.0

    else:

        effective_ml_weight = 0.0
        effective_tax_weight = 0.0

    combined = {}

    for track in TRACKS:

        combined[track] = (

            effective_ml_weight
            * ml_scores[track]

            +

            effective_tax_weight
            * tax_scores[track]

        )

    # Normalise combined scores.

    total = sum(combined.values())

    if total > 0:

        combined = {

            track: score / total

            for track, score in combined.items()

        }

    return {

        "combined": combined,

        "ml": ml_scores,

        "taxonomy": tax_scores,

        "model_known": model_known,

        "taxonomy_only": taxonomy_only,

        "effective_ml_weight": effective_ml_weight,

        "effective_taxonomy_weight": (
            effective_tax_weight
        )

    }


# ============================================================
# 7. GENERATE CAREER RECOMMENDATIONS
# ============================================================

def recommend_careers(
    user_input,
    bundle,
    top_n=3
):

    recognition = recognise_student_skills(
        user_input
    )

    recognised = recognition[
        "recognised"
    ]

    if not recognised:

        return {

            "success": False,

            "message": (
                "No recognised technical skills "
                "were entered."
            ),

            "recognition": recognition,

            "recommendations": []

        }

    model = bundle["model"]

    scores = hybrid_scores(
        recognised,
        model
    )

    ranked = sorted(

        scores["combined"].items(),

        key=lambda item: item[1],

        reverse=True

    )

    recommendations = []

    for track, score in ranked[:top_n]:

        if score <= 0:
            continue

        recommendations.append({

            "specialisation": track,

            "score": float(score),

            "ml_score": float(
                scores["ml"][track]
            ),

            "taxonomy_score": float(
                scores["taxonomy"][track]
            )

        })

    return {

        "success": len(recommendations) > 0,

        "recognition": recognition,

        "recommendations": recommendations,

        "model_known": scores[
            "model_known"
        ],

        "taxonomy_only": scores[
            "taxonomy_only"
        ],

        "effective_ml_weight": scores[
            "effective_ml_weight"
        ],

        "effective_taxonomy_weight": scores[
            "effective_taxonomy_weight"
        ]

    }


# ============================================================
# 8. TECHNICAL SKILLS GAP ANALYSIS
# ============================================================

def calculate_skill_gap(
    student_skills,
    target_specialisation,
    bundle,
    level="All Levels"
):

    target_specialisation = canonical_track(
        target_specialisation
    )

    if level == "Entry/Junior":

        profiles = bundle.get(
            "junior_profiles",
            {}
        )

    else:

        profiles = bundle.get(
            "demand_profiles",
            {}
        )

    profile = profiles.get(
        target_specialisation
    )

    # Do not silently use an All Levels profile
    # when an Entry/Junior profile is unavailable.

    if not profile or not profile.get("benchmark"):

        return {

            "success": False,

            "message": (
                "No usable benchmark available "
                "for this specialisation and level."
            )

        }

    benchmark = set(
        profile["benchmark"]
    )

    student_set = set(
        student_skills
    )

    matched = sorted(
        student_set & benchmark
    )

    missing = sorted(
        benchmark - student_set
    )

    outside_benchmark = sorted(
        student_set - benchmark
    )

    # Recognised skills relevant to the selected
    # specialisation, even if absent from benchmark.

    relevant_outside = []

    other_outside = []

    for skill in outside_benchmark:

        mappings = SKILL_SPECIALISATIONS.get(
            skill,
            {}
        )

        if target_specialisation in mappings:

            relevant_outside.append(skill)

        else:

            other_outside.append(skill)

    if benchmark:

        coverage = (
            len(matched)
            / len(benchmark)
        ) * 100

    else:

        coverage = 0.0

    counts = profile.get(
        "counts",
        {}
    )

    total_postings = profile.get(
        "total_postings",
        0
    )

    ranked_missing = sorted(

        missing,

        key=lambda skill: (
            -counts.get(skill, 0),
            skill
        )

    )

    roadmap = []

    for position, skill in enumerate(
        ranked_missing,
        start=1
    ):

        count = counts.get(
            skill,
            0
        )

        if total_postings > 0:

            frequency = (
                count / total_postings
            ) * 100

        else:

            frequency = 0.0

        roadmap.append({

            "priority": position,

            "skill": skill,

            "posting_count": count,

            "posting_percentage": frequency

        })

    return {

        "success": True,

        "specialisation": target_specialisation,

        "level": level,

        "coverage": coverage,

        "gap": 100 - coverage,

        "matched": matched,

        "missing": missing,

        "relevant_outside_benchmark": (
            relevant_outside
        ),

        "other_outside_benchmark": (
            other_outside
        ),

        "benchmark": sorted(benchmark),

        "total_postings": total_postings,

        "roadmap": roadmap

    }


# ============================================================
# 9. EXPLAIN TAXONOMY CONTRIBUTIONS
# ============================================================

def explain_taxonomy(
    skills,
    specialisation
):

    specialisation = canonical_track(
        specialisation
    )

    explanations = []

    for skill in skills:

        mappings = SKILL_SPECIALISATIONS.get(
            skill,
            {}
        )

        weight = mappings.get(
            specialisation,
            0.0
        )

        if weight > 0:

            explanations.append({

                "skill": skill,

                "relevance_weight": weight

            })

    explanations = sorted(

        explanations,

        key=lambda item: (
            -item["relevance_weight"],
            item["skill"]
        )

    )

    return explanations


# ============================================================
# 10. EXPLAIN MODEL CONTRIBUTIONS
# ============================================================

def explain_ml(
    skills,
    specialisation,
    model
):

    specialisation = canonical_track(
        specialisation
    )

    classifier = model.named_steps[
        "classifier"
    ]

    vectorizer = model.named_steps[
        "vectorizer"
    ]

    # Coefficient explanations are available
    # only for Logistic Regression.

    if not hasattr(
        classifier,
        "coef_"
    ):

        return []

    classes = list(
        classifier.classes_
    )

    if specialisation not in classes:

        return []

    class_index = classes.index(
        specialisation
    )

    # Multiclass logistic regression normally has
    # one coefficient row per class.
    # Binary logistic regression has one row.

    coefficients = classifier.coef_

    if len(classes) == 2:

        if class_index == 1:

            weights = coefficients[0]

        else:

            weights = -coefficients[0]

    else:

        weights = coefficients[
            class_index
        ]

    contributions = []

    for skill in skills:

        if skill not in vectorizer.vocabulary_:
            continue

        feature_index = (
            vectorizer.vocabulary_[skill]
        )

        weight = float(
            weights[feature_index]
        )

        contributions.append({

            "skill": skill,

            "coefficient": weight

        })

    contributions = sorted(

        contributions,

        key=lambda item: (
            -item["coefficient"],
            item["skill"]
        )

    )

    return contributions


# ============================================================
# 11. COMPLETE STUDENT ANALYSIS
# ============================================================

def analyse_student(
    user_input,
    bundle,
    target_specialisation=None,
    level="All Levels"
):

    recommendations = recommend_careers(
        user_input,
        bundle
    )

    if not recommendations["success"]:

        return recommendations

    recognised = recommendations[
        "recognition"
    ]["recognised"]

    primary_track = recommendations[
        "recommendations"
    ][0]["specialisation"]

    if target_specialisation:

        selected_track = canonical_track(
            target_specialisation
        )

    else:

        selected_track = primary_track

    gap_analysis = calculate_skill_gap(
        recognised,
        selected_track,
        bundle,
        level
    )

    taxonomy_explanation = explain_taxonomy(
        recognised,
        selected_track
    )

    ml_explanation = explain_ml(
        recommendations["model_known"],
        selected_track,
        bundle["model"]
    )

    return {

        "success": True,

        "recognition": recommendations[
            "recognition"
        ],

        "recommendations": recommendations[
            "recommendations"
        ],

        "primary_specialisation": primary_track,

        "selected_specialisation": selected_track,

        "model_known": recommendations[
            "model_known"
        ],

        "taxonomy_only": recommendations[
            "taxonomy_only"
        ],

        "effective_ml_weight": recommendations[
            "effective_ml_weight"
        ],

        "effective_taxonomy_weight": recommendations[
            "effective_taxonomy_weight"
        ],

        "gap_analysis": gap_analysis,

        "taxonomy_explanation": (
            taxonomy_explanation
        ),

        "ml_explanation": ml_explanation

    }


# ============================================================
# 12. TEST THE RECOMMENDER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING IT CAREER RECOMMENDER")
    print("=" * 60)

    bundle = load_recommender()

    test_cases = [

        "Cisco Packet Tracer",

        "Data Modelling, Data Visualisation, Machine Learning",

        "Python, SQL, Power BI",

        "Docker, Kubernetes, Terraform",

        "Wireshark, Nmap, Network Security",

        "Unknown Technology Example"

    ]

    for example in test_cases:

        print("\n" + "-" * 60)

        print("INPUT:", example)

        result = analyse_student(
            example,
            bundle
        )

        if not result["success"]:

            print(
                result.get(
                    "message",
                    "No recommendation available."
                )
            )

            continue

        print(
            "Recognised:",
            result["recognition"]["recognised"]
        )

        print(
            "Known to ML:",
            result["model_known"]
        )

        print(
            "Taxonomy only:",
            result["taxonomy_only"]
        )

        print(
            "Primary recommendation:",
            result["primary_specialisation"]
        )

        print("\nTop recommendations:")

        for item in result["recommendations"]:

            print(
                item["specialisation"],
                "-",
                round(item["score"] * 100, 1),
                "%"
            )

        gap = result["gap_analysis"]

        if gap["success"]:

            print(
                "Benchmark coverage:",
                round(gap["coverage"], 1),
                "%"
            )

            print(
                "Relevant skills outside benchmark:",
                gap["relevant_outside_benchmark"]
            )

    print("\nTesting complete.")


# In[ ]:




