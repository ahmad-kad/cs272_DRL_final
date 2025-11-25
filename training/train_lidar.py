from training.trainer_core import CurriculumTrainer
import wandb

def run_lidar_curriculum(timesteps_per_phase=30000):
    trainer = CurriculumTrainer(modality="lidar")
    
    # === PHASE 1: HIGHWAY FOUNDATION ===
    # Learn basic lane keeping and speed control
    model = trainer.train_phase(
        phase_name="1_highway_easy", 
        env_name="highway-v0", 
        timesteps=timesteps_per_phase, 
        difficulty="easy"
    )
    
    # === PHASE 2: MERGE MASTERY ===
    # Learn to handle merging vehicles (interaction)
    model = trainer.train_phase(
        phase_name="2_merge_medium", 
        env_name="merge-v0", 
        timesteps=timesteps_per_phase, 
        model=model, 
        difficulty="medium"
    )
    
    # === PHASE 3: INTERSECTION GENERALIZATION ===
    # The hardest challenge - fine-tuning for complex interactions
    # Using a higher difficulty and potentially more steps if needed
    model = trainer.train_phase(
        phase_name="3_intersection_hard", 
        env_name="intersection-v0", 
        timesteps=timesteps_per_phase * 2, # Give more time for the hardest task
        model=model, 
        difficulty="hard"
    )
    
    print("Lidar Curriculum Complete!")
    return model

