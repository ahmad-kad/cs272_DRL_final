import unittest
import os
import shutil
from training.trainer_core import CurriculumTrainer
from training.train_contrastive import ContrastiveTrainer
from utils.config import get_curriculum_config

class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = "outputs_test"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_config_factory(self):
        """Ensure config factory returns valid configs for all envs/difficulties"""
        for env in ["highway-v0", "merge-v0", "intersection-v0"]:
            for diff in ["easy", "medium", "hard"]:
                cfg = get_curriculum_config(env, diff, "lidar")
                self.assertIn("observation", cfg)
                self.assertIn("vehicles_count", cfg)

    def test_lidar_training_loop(self):
        """Ensure trainer can initialize and run a tiny step without crashing"""
        trainer = CurriculumTrainer(modality="lidar", base_dir=self.test_dir)
        try:
            # Run for small steps to verify loop
            model = trainer.train_phase("test_phase", "highway-v0", timesteps=100, difficulty="easy", adaptive=False)
            
            # Test Contrastive fine-tuning initiation
            # Use the model just trained as base
            contrastive = ContrastiveTrainer(
                base_model_path=os.path.join(self.test_dir, "models", "lidar_test_phase_final.zip"), 
                modality="lidar", 
                base_dir=self.test_dir
            )
            contrastive.train(timesteps=100, difficulty="hard")
            
        except Exception as e:
            self.fail(f"Training crashed: {e}")

if __name__ == '__main__':
    unittest.main()
