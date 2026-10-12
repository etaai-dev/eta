//! Reinforcement learning environment interfaces.

/// The result of one environment transition.
#[derive(Debug, Clone, PartialEq)]
pub struct Step<O> {
  /// Observation after the action.
  pub observation: O,
  /// Immediate reward from the action.
  pub reward: f64,
  /// Whether the environment reached a terminal state.
  pub terminated: bool,
  /// Whether an external limit, such as a time limit, ended the episode.
  pub truncated: bool,
}

/// An environment driven by agent actions.
pub trait Environment {
  /// Observation exposed to the agent.
  type Observation;
  /// Action accepted by the environment.
  type Action;
  /// Errors produced by environment interaction.
  type Error: std::error::Error;

  /// Starts an episode, optionally resetting the random seed.
  ///
  /// # Errors
  /// Returns the implementation's error if the environment cannot reset.
  fn reset(&mut self, seed: Option<u64>) -> Result<Self::Observation, Self::Error>;

  /// Applies an action. Reset after a terminated or truncated transition.
  ///
  /// # Errors
  /// Returns the implementation's error for invalid actions or episode state.
  fn step(&mut self, action: &Self::Action) -> Result<Step<Self::Observation>, Self::Error>;
}
