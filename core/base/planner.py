# planner.py
from typing import List, Dict, Optional, Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
import json

class AgentPlanner:
    """
    Planning module that generates and monitors execution plans.
    Supports both supervised and autonomous modes.
    """
    
    def __init__(self, model, mode: Literal["supervised", "autonomous"] = "supervised"):
        self.model = model
        self.mode = mode
        self.current_plan = []
        self.execution_state = {}
        self.completed_steps = []
        
    def generate_plan(self, user_goal: str, context: Dict) -> List[Dict]:
        """
        Generate a dynamic plan based on user goal and context.
        
        Args:
            user_goal: High-level goal from user
            context: Current system context (available agents, data paths, etc.)
        
        Returns:
            List of plan steps with agent assignments and parameters
        """
        planning_prompt = self._build_planning_prompt(user_goal, context)
        
        response = self.model.invoke([
            SystemMessage(content=planning_prompt),
            HumanMessage(content=f"User Goal: {user_goal}\n\nGenerate a detailed execution plan.")
        ])
        
        # Parse plan from LLM response
        plan = self._parse_plan(response.content)
        self.current_plan = plan
        return plan
    
    def _build_planning_prompt(self, user_goal: str, context: Dict) -> str:
        """Build mode-specific planning prompt"""
        
        base_prompt = """You are an intelligent planner for an Interp data analysis system.

Available Agents:
- FeatureFinder: Handles feature extraction from language model activations
- FeatureExplainer: Explains SAE features with hypothesis/test/refine loop

Your task is to generate a structured execution plan to achieve the user's goal.

Output your plan as a JSON array with the following structure:
[
    {
        "step": 1,
        "agent": "FeatureFinder",
        "task": "Run spike sorting pipeline",
        "dependencies": [],
        "parameters_needed": ["raw_data_path", "save_path"],
        "success_criteria": "Curated sorting analyzer saved",
        "estimated_duration": "15-30 minutes"
    },
    ...
]

Consider:
1. What data is available or needed
2. What preprocessing steps are required
3. Dependencies between steps
4. Whether steps can run in parallel
5. What outputs each step produces
"""
        
        if self.mode == "supervised":
            mode_specific = """
Mode: SUPERVISED
- Plan should include user confirmation points after major steps
- Be conservative with parameter choices
- Explicitly state when user input is needed
- Ask for guidance on ambiguous decisions
"""
        else:  # autonomous
            mode_specific = """
Mode: AUTONOMOUS
- Plan should minimize user confirmations
- Use intelligent defaults based on data characteristics
- Include self-evaluation checkpoints
- Only escalate to user on critical uncertainties or failures
- Provide confidence scores for decisions
"""
        
        return base_prompt + mode_specific
    
    def _parse_plan(self, llm_response: str) -> List[Dict]:
        """Parse plan from LLM response"""
        try:
            # Try to extract JSON from response
            start_idx = llm_response.find('[')
            end_idx = llm_response.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = llm_response[start_idx:end_idx]
                plan = json.loads(json_str)
                return plan
        except Exception as e:
            print(f"Failed to parse plan JSON: {e}")
            
        # Fallback: simple parsing
        return []
    
    def should_proceed(self, current_step: Dict, execution_result: Dict) -> Dict:
        """
        Decide whether to proceed to next step based on current results.
        
        Returns:
            Dict with keys: 'decision', 'reason', 'confidence'
        """
        if self.mode == "supervised":
            # Always require user confirmation in supervised mode
            return {
                'decision': 'ask_user',
                'reason': 'Supervised mode requires user confirmation',
                'confidence': 1.0
            }
        
        # Autonomous mode: evaluate quality and decide
        evaluation_prompt = f"""You are evaluating the execution of a data analysis step.

Step: {current_step}
Result: {execution_result}

Assess:
1. Was the step completed successfully?
2. Does the output meet quality criteria?
3. Are there any issues or anomalies?
4. Confidence in the results (0-1)

Respond with JSON:
{{
    "success": true/false,
    "quality_score": 0-1,
    "issues": ["list", "of", "issues"],
    "confidence": 0-1,
    "recommendation": "proceed" | "retry" | "escalate",
    "reasoning": "explanation"
}}
"""
        
        response = self.model.invoke([
            SystemMessage(content=evaluation_prompt),
            HumanMessage(content="Evaluate the step execution.")
        ])
        
        try:
            evaluation = self._parse_evaluation(response.content)
            
            if evaluation['confidence'] < 0.5 or not evaluation['success']:
                return {
                    'decision': 'escalate',
                    'reason': evaluation['reasoning'],
                    'confidence': evaluation['confidence']
                }
            elif evaluation['quality_score'] > 0.7:
                return {
                    'decision': 'proceed',
                    'reason': evaluation['reasoning'],
                    'confidence': evaluation['confidence']
                }
            else:
                return {
                    'decision': 'retry',
                    'reason': evaluation['reasoning'],
                    'confidence': evaluation['confidence']
                }
        except Exception as e:
            # On error, escalate to human
            return {
                'decision': 'escalate',
                'reason': f'Evaluation failed: {e}',
                'confidence': 0.0
            }
    
    def _parse_evaluation(self, llm_response: str) -> Dict:
        """Parse evaluation response"""
        try:
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = llm_response[start_idx:end_idx]
                evaluation = json.loads(json_str)
                return evaluation
        except Exception as e:
            print(f"Failed to parse evaluation JSON: {e}")
        
        # Return default conservative evaluation
        return {
            'success': False,
            'quality_score': 0.5,
            'issues': ['Could not parse evaluation'],
            'confidence': 0.3,
            'recommendation': 'escalate',
            'reasoning': 'Evaluation parsing failed'
        }
    
    def adapt_plan(self, issue: str, current_step_idx: int) -> List[Dict]:
        """
        Adapt plan when issues occur (autonomous mode)
        
        Args:
            issue: Description of the issue encountered
            current_step_idx: Index of step where issue occurred
        
        Returns:
            Updated plan
        """
        if self.mode == "supervised":
            # In supervised mode, just flag for user
            return self.current_plan
        
        adaptation_prompt = f"""An issue occurred during plan execution:

Original Plan: {json.dumps(self.current_plan, indent=2)}
Current Step: {current_step_idx}
Issue: {issue}
Completed Steps: {self.completed_steps}

Generate an adapted plan that addresses the issue. You can:
1. Retry the current step with different parameters
2. Add diagnostic/debugging steps
3. Skip the problematic step if it's optional
4. Modify downstream steps to account for the issue

Return the updated plan in the same JSON format.
"""
        
        response = self.model.invoke([
            SystemMessage(content=adaptation_prompt),
            HumanMessage(content="Generate adapted plan.")
        ])
        
        adapted_plan = self._parse_plan(response.content)
        if adapted_plan:
            self.current_plan = adapted_plan
            return adapted_plan
        else:
            return self.current_plan
    
    def get_next_step(self) -> Optional[Dict]:
        """Get next step to execute"""
        if not self.current_plan:
            return None
        
        for step in self.current_plan:
            if step['step'] not in [s['step'] for s in self.completed_steps]:
                # Check if dependencies are met
                deps_met = all(
                    dep in [s['step'] for s in self.completed_steps]
                    for dep in step.get('dependencies', [])
                )
                if deps_met:
                    return step
        
        return None
    
    def mark_step_complete(self, step: Dict, result: Dict):
        """Mark a step as completed"""
        self.completed_steps.append({
            'step': step['step'],
            'agent': step['agent'],
            'result': result,
            'timestamp': None  # Could add timestamp
        })
        self.execution_state[step['step']] = result

