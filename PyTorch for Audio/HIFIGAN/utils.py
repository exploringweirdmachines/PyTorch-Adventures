def print_log(log, iter):
    print("=" * 40)
    print("{:^40}".format(f"Training Log Summary: Iteration {iter}"))
    print("=" * 40)
    
    for key, value in log.items():
        if isinstance(value, list) and value:
            display_value = value[-1]  # show the latest entry
        else:
            display_value = value if not isinstance(value, list) else "N/A"
        
        print(f"{key:<20}: {display_value:.4f}" if isinstance(display_value, (int, float)) else f"{key:<20}: {display_value}")
    
    print("=" * 40)