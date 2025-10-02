import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Set, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ICLCR:
    def __init__(self, 
                 llm_model,
                 k_demonstrations: int = 10,
                 mi_threshold: float = 0.1,
                 selection_strategy: str = 'weighted_knn',
                 embedding_model = None):

        self.llm_model = llm_model
        self.k = k_demonstrations
        self.beta = mi_threshold
        self.selection_strategy = selection_strategy
        self.embedding_model = embedding_model
    
    def compute_mutual_information(self, 
                                   X: pd.Series, 
                                   Y: pd.Series) -> float:
        # Remove missing values
        mask = (~X.isna()) & (~Y.isna())
        X_clean = X[mask]
        Y_clean = Y[mask]
        
        if len(X_clean) == 0:
            return 0.0
        
        # Discretize numerical attributes
        if pd.api.types.is_numeric_dtype(X_clean):
            try:
                kbd = KBinsDiscretizer(n_bins=min(10, len(X_clean.unique())), 
                                      encode='ordinal', 
                                      strategy='quantile')
                X_clean = kbd.fit_transform(X_clean.values.reshape(-1, 1)).flatten()
            except:
                pass
        
        # Convert to discrete values
        X_vals = X_clean.astype(str)
        Y_vals = Y_clean.astype(str)
        
        # Compute joint and marginal probabilities
        joint_counts = pd.crosstab(X_vals, Y_vals)
        joint_probs = joint_counts / joint_counts.sum().sum()
        
        X_probs = joint_probs.sum(axis=1)
        Y_probs = joint_probs.sum(axis=0)
        
        # Compute mutual information
        mi = 0.0
        for i in joint_probs.index:
            for j in joint_probs.columns:
                if joint_probs.loc[i, j] > 0:
                    mi += joint_probs.loc[i, j] * np.log(
                        joint_probs.loc[i, j] / (X_probs[i] * Y_probs[j])
                    )
        
        return max(0.0, mi)
    
    def compress_demonstration(self, 
                              tuple_data: Dict, 
                              target_attr: str,
                              table: pd.DataFrame) -> Dict:
        """
        Compress demonstration example by removing non-relevant attributes.
        
        Args:
            tuple_data: Dictionary of attribute values
            target_attr: Target attribute for prediction
            table: Full table for computing mutual information
            
        Returns:
            Compressed tuple data
        """
        compressed = {}
        
        for attr, value in tuple_data.items():
            # Skip missing values and target attribute
            if pd.isna(value) or attr == target_attr:
                continue
            
            # Compute mutual information with target attribute
            mi = self.compute_mutual_information(
                table[attr], 
                table[target_attr]
            )
            
            # Include attribute only if MI exceeds threshold
            if mi >= self.beta:
                compressed[attr] = value
        
        return compressed
    
    def create_prompt_template(self, 
                              demonstrations: List[Tuple[Dict, str]],
                              target_tuple: Dict,
                              target_attr: str) -> str:
        """
        Create prompt input for LLM.
        
        Args:
            demonstrations: List of (compressed_tuple, correct_value) pairs
            target_tuple: Target tuple to predict
            target_attr: Target attribute name
            
        Returns:
            Formatted prompt string
        """
        prompt = "Given the following examples, predict the correct value.\n\n"
        
        # Add demonstration examples
        for i, (demo_tuple, correct_value) in enumerate(demonstrations, 1):
            # Build "if" clause
            conditions = []
            for attr, value in demo_tuple.items():
                conditions.append(f"the attribute '{attr}' is '{value}'")
            
            if_clause = ", ".join(conditions)
            
            # Build "then" clause
            prompt += f"Example {i}: If {if_clause}, "
            prompt += f"then the value of attribute '{target_attr}' should be '{correct_value}'.\n\n"
        
        # Add target question
        conditions = []
        for attr, value in target_tuple.items():
            if not pd.isna(value):
                conditions.append(f"the attribute '{attr}' is '{value}'")
        
        if_clause = ", ".join(conditions)
        prompt += f"Question: If {if_clause}, "
        prompt += f"what should the value of attribute '{target_attr}' be?\n\n"
        prompt += "Answer:"
        
        return prompt
    
    def encode_tuple(self, tuple_data: Dict) -> np.ndarray:
        """
        Encode tuple into embedding vector.
        
        Args:
            tuple_data: Dictionary of attribute values
            
        Returns:
            Embedding vector
        """
        if self.embedding_model is None:
            # Simple bag-of-words encoding
            text = " ".join([str(v) for v in tuple_data.values() if not pd.isna(v)])
            # Use a simple hash-based encoding for demonstration
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=100)
            return vectorizer.fit_transform([text]).toarray()[0]
        else:
            return self.embedding_model.encode(tuple_data)
    
    def select_demonstrations_random(self, 
                                    table: pd.DataFrame,
                                    target_attr: str,
                                    k: int) -> List[Tuple[Dict, str]]:
        """Random demonstration selection."""
        # Filter rows with non-null target attribute
        valid_rows = table[table[target_attr].notna()]
        
        if len(valid_rows) < k:
            k = len(valid_rows)
        
        selected = valid_rows.sample(n=k)
        demonstrations = []
        
        for _, row in selected.iterrows():
            tuple_data = row.to_dict()
            correct_value = tuple_data[target_attr]
            compressed = self.compress_demonstration(tuple_data, target_attr, table)
            demonstrations.append((compressed, correct_value))
        
        return demonstrations
    
    def select_demonstrations_knn(self,
                                 table: pd.DataFrame,
                                 target_tuple: Dict,
                                 target_attr: str,
                                 k: int) -> List[Tuple[Dict, str]]:
        """k-NN demonstration selection based on cosine similarity."""
        # Filter rows with non-null target attribute
        valid_rows = table[table[target_attr].notna()]
        
        if len(valid_rows) == 0:
            return []
        
        # Encode target tuple
        target_emb = self.encode_tuple(target_tuple)
        
        # Encode all valid tuples
        embeddings = []
        for _, row in valid_rows.iterrows():
            emb = self.encode_tuple(row.to_dict())
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # Compute cosine similarity
        similarities = cosine_similarity([target_emb], embeddings)[0]
        
        # Select top-k most similar
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        demonstrations = []
        for idx in top_k_indices:
            row = valid_rows.iloc[idx]
            tuple_data = row.to_dict()
            correct_value = tuple_data[target_attr]
            compressed = self.compress_demonstration(tuple_data, target_attr, table)
            demonstrations.append((compressed, correct_value))
        
        return demonstrations
    
    def select_demonstrations_weighted_knn(self,
                                          table: pd.DataFrame,
                                          target_tuple: Dict,
                                          target_attr: str,
                                          k: int) -> List[Tuple[Dict, str]]:
        """Weighted k-NN using mutual information weights."""
        # Filter rows with non-null target attribute
        valid_rows = table[table[target_attr].notna()]
        
        if len(valid_rows) == 0:
            return []
        
        # Compute MI weights for each attribute
        mi_weights = {}
        for attr in target_tuple.keys():
            if attr != target_attr and attr in table.columns:
                mi = self.compute_mutual_information(table[attr], table[target_attr])
                mi_weights[attr] = mi
        
        # Normalize weights
        total_weight = sum(mi_weights.values())
        if total_weight > 0:
            mi_weights = {k: v/total_weight for k, v in mi_weights.items()}
        
        # Compute weighted similarity
        similarities = []
        for _, row in valid_rows.iterrows():
            sim = 0.0
            for attr, weight in mi_weights.items():
                if attr in target_tuple and attr in row.index:
                    if target_tuple[attr] == row[attr]:
                        sim += weight
            similarities.append(sim)
        
        # Select top-k
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        demonstrations = []
        for idx in top_k_indices:
            row = valid_rows.iloc[idx]
            tuple_data = row.to_dict()
            correct_value = tuple_data[target_attr]
            compressed = self.compress_demonstration(tuple_data, target_attr, table)
            demonstrations.append((compressed, correct_value))
        
        return demonstrations
    
    def select_demonstrations(self,
                            table: pd.DataFrame,
                            target_tuple: Dict,
                            target_attr: str) -> List[Tuple[Dict, str]]:
        """Select demonstration examples based on strategy."""
        if self.selection_strategy == 'random':
            return self.select_demonstrations_random(table, target_attr, self.k)
        elif self.selection_strategy == 'knn':
            return self.select_demonstrations_knn(table, target_tuple, target_attr, self.k)
        elif self.selection_strategy == 'weighted_knn':
            return self.select_demonstrations_weighted_knn(table, target_tuple, target_attr, self.k)
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
    
    def resolve_conflict(self,
                        integrable_set: pd.DataFrame,
                        target_attr: str,
                        table: pd.DataFrame) -> str:
        """
        Resolve conflict for a specific attribute in an integrable set.
        
        Args:
            integrable_set: DataFrame containing tuples in the integrable set
            target_attr: Attribute with conflicting values
            table: Full table for selecting demonstrations
            
        Returns:
            Predicted correct value
        """
        # Get candidate values (non-null values in the integrable set)
        candidates = integrable_set[target_attr].dropna().unique()
        
        if len(candidates) == 0:
            return None
        if len(candidates) == 1:
            return candidates[0]
        
        # Create target tuple by aggregating non-null values
        target_tuple = {}
        for col in integrable_set.columns:
            non_null = integrable_set[col].dropna()
            if len(non_null) > 0:
                # Use most frequent value
                target_tuple[col] = non_null.mode()[0] if len(non_null.mode()) > 0 else non_null.iloc[0]
        
        # Select demonstration examples
        demonstrations = self.select_demonstrations(table, target_tuple, target_attr)
        
        # Compress target tuple
        compressed_target = self.compress_demonstration(target_tuple, target_attr, table)
        
        # Create prompt
        prompt = self.create_prompt_template(demonstrations, compressed_target, target_attr)
        
        # Get LLM prediction
        predicted_value = self.llm_model.predict(prompt, candidates)
        
        return predicted_value
    
    def resolve_integrable_set(self,
                              integrable_set: pd.DataFrame,
                              table: pd.DataFrame) -> Dict:
        """
        Resolve all conflicts in an integrable set.
        
        Args:
            integrable_set: DataFrame containing tuples in the integrable set
            table: Full table for selecting demonstrations
            
        Returns:
            Dictionary of resolved attribute values
        """
        resolved_tuple = {}
        
        for attr in integrable_set.columns:
            # Get unique non-null values
            unique_values = integrable_set[attr].dropna().unique()
            
            if len(unique_values) == 0:
                # Missing value attribute
                resolved_tuple[attr] = None
            elif len(unique_values) == 1:
                # Unique value attribute
                resolved_tuple[attr] = unique_values[0]
            else:
                # Multiple value attribute - resolve conflict
                resolved_tuple[attr] = self.resolve_conflict(
                    integrable_set, attr, table
                )
        
        return resolved_tuple


# Mock LLM class for demonstration
class MockLLM:
    """Mock LLM for demonstration purposes."""
    
    def predict(self, prompt: str, candidates: List) -> str:
        """
        Predict the most likely candidate.
        For a real implementation, this would call an actual LLM API.
        """
        # Simple heuristic: return most frequent candidate mentioned in prompt
        candidate_counts = {c: prompt.lower().count(str(c).lower()) for c in candidates}
        return max(candidate_counts, key=candidate_counts.get)
