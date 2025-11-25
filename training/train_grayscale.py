from training.trainer_core import CurriculumTrainer
import wandb

def run_grayscale_curriculum(timesteps_per_phase=30000):
    trainer = CurriculumTrainer(modality="grayscale")

    # === PHASE 1: HIGHWAY FOUNDATION ===
    # Learn basic lane keeping and speed control with visual input
    model = trainer.train_phase(
        phase_name="1_highway_easy",
        env_name="highway-v0",
        timesteps=timesteps_per_phase,
        difficulty="easy"
    )

    # === PHASE 2: MERGE MASTERY ===
    # Learn to handle merging vehicles with visual perception
    model = trainer.train_phase(
        phase_name="2_merge_medium",
        env_name="merge-v0",
        timesteps=timesteps_per_phase,
        model=model,
        difficulty="medium"
    )

    # === PHASE 3: INTERSECTION GENERALIZATION ===
    # The hardest challenge - complex interactions with visual input
    model = trainer.train_phase(
        phase_name="3_intersection_hard",
        env_name="intersection-v0",
        timesteps=timesteps_per_phase * 2,  # Give more time for visual learning
        model=model,
        difficulty="hard"
    )

    print("Grayscale Curriculum Complete!")
    return model
