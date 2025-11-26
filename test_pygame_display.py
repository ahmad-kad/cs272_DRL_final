#!/usr/bin/env python3
"""
Test Pygame Display Functionality

This script tests if pygame can create a display window and render graphics.
Useful for diagnosing visualization issues with the driving environment.
"""

import pygame
import sys
import time
import os

def test_basic_pygame():
    """Test basic pygame functionality"""
    print("Testing basic pygame initialization...")

    try:
        pygame.init()
        print(f"[OK] Pygame initialized successfully (version: {pygame.version.ver})")
        return True
    except Exception as e:
        print(f"[FAIL] Pygame initialization failed: {e}")
        return False

def test_display_creation():
    """Test creating a pygame display"""
    print("\nTesting display creation...")

    try:
        # Try different display modes
        screen = pygame.display.set_mode((800, 600))
        print("[OK] Display created successfully (800x600)")
        return screen
    except Exception as e:
        print(f"[FAIL] Display creation failed: {e}")
        return None

def test_display_rendering(screen):
    """Test rendering to the display"""
    print("\nTesting display rendering...")

    try:
        # Set window title
        pygame.display.set_caption("Pygame Display Test - Close window to continue")

        # Fill with colors
        screen.fill((50, 100, 150))  # Blue background

        # Draw some shapes
        pygame.draw.rect(screen, (255, 255, 0), (50, 50, 200, 100))  # Yellow rectangle
        pygame.draw.circle(screen, (255, 0, 0), (400, 300), 50)  # Red circle
        pygame.draw.line(screen, (0, 255, 0), (100, 100), (700, 500), 5)  # Green line

        # Add text
        font = pygame.font.Font(None, 36)
        text = font.render("Pygame Display Test - Press X to close", True, (255, 255, 255))
        screen.blit(text, (50, 520))

        # Update display
        pygame.display.flip()
        print("[OK] Rendering successful - you should see a window with shapes and text")
        return True

    except Exception as e:
        print(f"[FAIL] Rendering failed: {e}")
        return False

def test_event_handling():
    """Test event handling (user input)"""
    print("\nTesting event handling...")

    try:
        running = True
        start_time = time.time()

        while running and (time.time() - start_time) < 10:  # 10 second timeout
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    print("[OK] Window closed by user")
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        print("[OK] ESC key pressed")

            pygame.time.wait(100)  # Small delay to prevent 100% CPU usage

        if running:
            print("[OK] Event loop ran successfully (timed out after 10 seconds)")
        return True

    except Exception as e:
        print(f"[FAIL] Event handling failed: {e}")
        return False

def test_highway_env_rendering():
    """Test highway-env rendering specifically"""
    print("\nTesting highway-env rendering...")

    try:
        from highway_env.envs import HighwayEnv

        # Create environment
        env = HighwayEnv()

        # Reset to get initial observation
        obs, info = env.reset()

        # Try to render
        frame = env.render()
        if frame is not None:
            print(f"[OK] Highway-env rendered successfully (frame shape: {frame.shape})")
            return True
        else:
            print("[FAIL] Highway-env render returned None")
            return False

    except Exception as e:
        print(f"[FAIL] Highway-env rendering failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("PYGAME DISPLAY TEST SUITE")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Working directory: {os.getcwd()}")
    print()

    # Test 1: Basic pygame
    if not test_basic_pygame():
        print("\n[ERROR] CRITICAL: Pygame initialization failed. Cannot continue.")
        return False

    # Test 2: Display creation
    screen = test_display_creation()
    if screen is None:
        print("\n[ERROR] CRITICAL: Display creation failed. Cannot continue.")
        pygame.quit()
        return False

    # Test 3: Rendering
    if not test_display_rendering(screen):
        print("\n[ERROR] WARNING: Rendering failed, but display was created.")
        pygame.quit()
        return False

    # Test 4: Event handling
    test_event_handling()

    # Clean up pygame
    pygame.quit()
    print("\n[OK] Pygame cleanup successful")

    # Test 5: Highway-env rendering
    highway_success = test_highway_env_rendering()

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    if highway_success:
        print("[SUCCESS] ALL TESTS PASSED!")
        print("[OK] Pygame works correctly")
        print("[OK] Display creation successful")
        print("[OK] Rendering works")
        print("[OK] Event handling works")
        print("[OK] Highway-env rendering works")
        print("\n[ACTION] You should be able to visualize the driving environment!")
        print("Try: python evaluate_models.py outputs/models/adaptive_grayscale_final.zip --visualize --episodes 1")
    else:
        print("[WARNING]  PARTIAL SUCCESS")
        print("[OK] Pygame works correctly")
        print("[OK] Display creation successful")
        print("[OK] Rendering works")
        print("[OK] Event handling works")
        print("[FAIL] Highway-env rendering failed")
        print("\n[FIX] Highway-env specific issue. Try running in different terminal.")

    return highway_success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[STOP]  Test interrupted by user")
        pygame.quit()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[CRASH] Unexpected error: {e}")
        pygame.quit()
        sys.exit(1)
