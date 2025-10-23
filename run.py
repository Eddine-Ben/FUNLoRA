import os
from train import parse_args, create_config_from_args, main

if __name__ == "__main__":
    try:
        args = parse_args()
        config = create_config_from_args(args)
        os.makedirs(config.get('output_dir', '.'), exist_ok=True)
        main(config)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user (Ctrl+C)")
        print("💡 You can resume training using --resume with the latest checkpoint")
        print("📁 Check for checkpoint files: checkpoint_epoch_*.pth")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("💡 Common fixes:")
        print("   - On Windows: Training uses num_workers=0 (no multiprocessing)")
        print("   - Check dataset path is correct")
        print("   - Ensure CUDA is available")
        import traceback
        traceback.print_exc()

