from model import SpamDetector

def run_interface():
    print("=================================================")
    print("       AI Scam detector(v1.0) Initialized     ")
    print("=================================================")
    ai = SpamDetector()
    print("System Ready! Type 'exit' to shut down.\n")
    
    ai.train("data/spam_data.txt")
    print("Training completed!\n")
    
    while True:
        user_input = input ("Enter a text message to scan: ")
        if user_input.lower() == 'exit':
            print("Shutting down. Goodbye")
            break
        if not user_input.strip(): 
            continue

        print ("\nCrunching mathematical layers...")
        results = ai.predict(user_input)

        print("----------------------------------------")
        print(f"1. Bayesian Evidence:        {results['bayes']}")
        print(f"2. Shannon Entropy Score:    {results['entropy']}")
        print(f"3. Vector Similarity Score:  {results['vector']}")
        print(f"   Raw Weight (z):           {results['z_raw_score']}")
        print("----------------------------------------")
        
        
        print(f"FINAL SCAM PROBABILITY:   {results['final_scam_probability']}%\n")

if __name__ == "__main__":
    run_interface()




