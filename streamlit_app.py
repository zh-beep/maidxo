import streamlit as st
import sys
import os
from typing import Dict, Any
import time
import json

# Add the Open-MAI-Dx-Orchestrator directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "Open-MAI-Dx-Orchestrator"))

try:
    from mai_dx import MaiDxOrchestrator
except ImportError:
    st.error("Could not import MaiDxOrchestrator. Please ensure the mai_dx package is installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="MAI-DxO: AI Physician Panel",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    st.title("MAI Diagnostic Orchestrator")
    st.write("AI-Powered Medical Diagnosis with Virtual Physician Panel")


    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        model_options = {
            "GPT-4o": "gpt-4o",
            "GPT-4": "gpt-4",
            "Gemini 2.5 Flash": "gemini/gemini-2.5-flash",
            "Gemini Pro": "gemini/gemini-pro",
            "Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",
        }
        
        selected_model_name = st.selectbox(
            "Select AI Model",
            list(model_options.keys()),
            index=0  # Default to GPT-4o
        )
        selected_model = model_options[selected_model_name]
        
        # Diagnostic mode
        diagnostic_modes = {
            "No Budget": "no_budget",
            "Budgeted": "budgeted", 
            "Question Only": "question_only",
            "Instant": "instant",
            "Ensemble": "ensemble"
        }
        
        selected_mode_name = st.selectbox(
            "Diagnostic Mode",
            list(diagnostic_modes.keys()),
            index=0
        )
        selected_mode = diagnostic_modes[selected_mode_name]
        
        # Budget settings (if budgeted mode)
        budget = None
        if selected_mode == "budgeted":
            budget = st.slider("Budget ($)", 1000, 10000, 3000, 500)
        
        # Iterations
        max_iterations = st.slider("Max Iterations", 1, 15, 5)
        
        # Ensemble settings
        num_ensemble_runs = 3
        if selected_mode == "ensemble":
            num_ensemble_runs = st.slider("Ensemble Runs", 2, 5, 3)

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Patient Case Information")
        
        # Tab layout for case input
        tab1, tab2, tab3 = st.tabs(["Initial Case", "Full Details", "Ground Truth"])
        
        with tab1:
            initial_case = st.text_area(
                "Initial Case Information",
                placeholder="Enter the initial patient presentation (chief complaint, brief history)...",
                height=150,
                value="A 29-year-old woman was admitted to the hospital because of sore throat and peritonsillar swelling and bleeding. Symptoms did not abate with antimicrobial therapy."
            )
        
        with tab2:
            full_case = st.text_area(
                "Complete Case Details",
                placeholder="Enter the full case details including history, physical exam, initial labs, imaging results...",
                height=300,
                value="""Patient: 29-year-old female.
History: Onset of sore throat 7 weeks prior to admission. Worsening right-sided pain and swelling.
No fevers, headaches, or gastrointestinal symptoms. Past medical history is unremarkable. No history of smoking or significant alcohol use.
Physical Exam: Right peritonsillar mass, displacing the uvula. No other significant findings.
Initial Labs: FBC, clotting studies normal.
MRI Neck: Showed a large, enhancing mass in the right peritonsillar space.
Biopsy (H&E): Infiltrative round-cell neoplasm with high nuclear-to-cytoplasmic ratio and frequent mitotic figures.
Biopsy (Immunohistochemistry for Carcinoma): CD31, D2-40, CD34, ERG, GLUT-1, pan-cytokeratin, CD45, CD20, CD3 all negative. Ki-67: 60% nuclear positivity.
Biopsy (Immunohistochemistry for Rhabdomyosarcoma): Desmin and MyoD1 diffusely positive. Myogenin multifocally positive.
Biopsy (FISH): No FOXO1 (13q14) rearrangements detected.
Final Diagnosis from Pathology: Embryonal rhabdomyosarcoma of the pharynx."""
            )
        
        with tab3:
            ground_truth = st.text_input(
                "Ground Truth Diagnosis (for evaluation)",
                placeholder="Enter the correct diagnosis for accuracy assessment...",
                value="Embryonal rhabdomyosarcoma of the pharynx"
            )
    
    with col2:
        st.subheader("Virtual Physician Panel")
        
        # Display the 8 agents
        agents_info = [
            ("Dr. Hypothesis", "Maintains differential diagnosis"),
            ("Dr. Test-Chooser", "Selects diagnostic tests"),
            ("Dr. Challenger", "Prevents cognitive biases"),
            ("Dr. Stewardship", "Ensures cost-effectiveness"),
            ("Dr. Checklist", "Quality control checks"),
            ("Consensus Coordinator", "Synthesizes decisions"),
            ("Gatekeeper", "Clinical information oracle"),
            ("Judge", "Evaluates accuracy")
        ]
        
        for name, description in agents_info:
            st.write(f"**{name}**: {description}")

    # Action buttons
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        run_diagnosis = st.button("Run Diagnosis", type="primary", use_container_width=True)
    
    with col_btn2:
        clear_results = st.button("Clear Results", use_container_width=True)
    
    # Initialize session state
    if 'diagnosis_results' not in st.session_state:
        st.session_state.diagnosis_results = None
    
    if clear_results:
        st.session_state.diagnosis_results = None
        st.rerun()

    # Run diagnosis
    if run_diagnosis:
        if not initial_case.strip():
            st.error("Please enter initial case information.")
            return
        
        if not full_case.strip():
            st.error("Please enter full case details.")
            return
        
        # Create orchestrator
        with st.spinner("Initializing MAI-DxO Virtual Physician Panel..."):
            try:
                if selected_mode == "budgeted" and budget:
                    orchestrator = MaiDxOrchestrator.create_variant(
                        selected_mode,
                        budget=budget,
                        model_name=selected_model,
                        max_iterations=max_iterations,
                    )
                else:
                    orchestrator = MaiDxOrchestrator.create_variant(
                        selected_mode,
                        model_name=selected_model,
                        max_iterations=max_iterations,
                    )
                
                # Run diagnosis
                with st.spinner("Virtual Physician Panel is analyzing the case..."):
                    start_time = time.time()
                    
                    if selected_mode == "ensemble":
                        result = orchestrator.run_ensemble(
                            initial_case_info=initial_case,
                            full_case_details=full_case,
                            ground_truth_diagnosis=ground_truth if ground_truth.strip() else None,
                            num_runs=num_ensemble_runs,
                        )
                    else:
                        result = orchestrator.run(
                            initial_case_info=initial_case,
                            full_case_details=full_case,
                            ground_truth_diagnosis=ground_truth if ground_truth.strip() else None,
                        )
                    
                    end_time = time.time()
                    result.processing_time = end_time - start_time
                    
                st.session_state.diagnosis_results = result
                st.success("Diagnosis complete!")
                
            except Exception as e:
                st.error(f"Error during diagnosis: {str(e)}")
                st.error("Please check your API keys in your .env file and model configuration.")
                return

    # Display results
    if st.session_state.diagnosis_results:
        result = st.session_state.diagnosis_results
        
        st.markdown("---")
        st.subheader("Diagnostic Results")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy Score", f"{result.accuracy_score}/5.0")
        
        with col2:
            st.metric("Total Cost", f"${result.total_cost:,.0f}")
        
        with col3:
            st.metric("Iterations", result.iterations)
        
        with col4:
            processing_time = getattr(result, 'processing_time', 0)
            st.metric("Processing Time", f"{processing_time:.1f}s")
        
        # Diagnosis details
        st.markdown("### Final Diagnosis")
        st.success(f"**{result.final_diagnosis}**")
        
        if ground_truth.strip():
            st.markdown("### Ground Truth Comparison")
            st.info(f"**Ground Truth:** {result.ground_truth}")
            
            if hasattr(result, 'accuracy_reasoning') and result.accuracy_reasoning:
                st.markdown("### Accuracy Assessment")
                st.write(result.accuracy_reasoning)
        
        # Show reasoning directly on page
        if hasattr(result, 'accuracy_reasoning') and result.accuracy_reasoning:
            st.markdown("### Diagnostic Reasoning")
            st.write(result.accuracy_reasoning)
        
        # Additional details if available
        if hasattr(result, 'diagnostic_reasoning') and result.diagnostic_reasoning:
            st.markdown("### Detailed Analysis")
            st.write(result.diagnostic_reasoning)
        
        if hasattr(result, 'tests_ordered') and result.tests_ordered:
            st.markdown("### Tests Ordered")
            for test in result.tests_ordered:
                st.write(f"- {test}")
        
        # Export results
        st.markdown("### Export Results")
        results_dict = {
            "final_diagnosis": result.final_diagnosis,
            "ground_truth": result.ground_truth,
            "accuracy_score": result.accuracy_score,
            "total_cost": result.total_cost,
            "iterations": result.iterations,
            "mode": selected_mode,
            "model": selected_model,
            "processing_time": getattr(result, 'processing_time', 0),
        }
        
        st.download_button(
            label="Download Results (JSON)",
            data=json.dumps(results_dict, indent=2),
            file_name=f"mai_dxo_results_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()