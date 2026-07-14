from __future__ import annotations

import math
import random
import copy
from statistics import mean, median

from neat.config import ConfigParameter, DefaultClassConfig
from neat.reproduction import DefaultReproduction


def epsilon_lexicase_select(members: list[tuple[int, object]]) -> tuple[int, object]:
    if not members:
        raise ValueError("Cannot select from an empty population.")
    case_count = len(getattr(members[0][1], "lexicase_scores", []))
    if case_count == 0:
        return random.choice(members)
    cases = list(range(case_count))
    random.shuffle(cases)
    candidates = list(members)
    for case in cases:
        values = [float(candidate.lexicase_scores[case]) for _, candidate in candidates]
        best = max(values)
        center = median(values)
        epsilon = median([abs(value - center) for value in values])
        survivors = [item for item in candidates if float(item[1].lexicase_scores[case]) >= best - epsilon]
        candidates = survivors or [max(candidates, key=lambda item: float(item[1].lexicase_scores[case]))]
        if len(candidates) == 1:
            break
    return random.choice(candidates)


class LexicaseReproduction(DefaultReproduction):
    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(
            param_dict,
            [
                ConfigParameter("elitism", int, 0),
                ConfigParameter("survival_threshold", float, 0.2),
                ConfigParameter("min_species_size", int, 1),
            ],
            "LexicaseReproduction",
        )

    def reproduce(self, config, species, pop_size, generation):
        config.genome_config.innovation_tracker = self.innovation_tracker
        self.innovation_tracker.reset_generation()

        all_fitnesses = []
        remaining_species = []
        for species_id, species_record, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(species_id, species_record)
            else:
                all_fitnesses.extend(member.fitness for member in species_record.members.values())
                remaining_species.append(species_record)
        if not remaining_species:
            species.species = {}
            return {}

        minimum = min(all_fitnesses)
        fitness_range = max(1.0, max(all_fitnesses) - minimum)
        for species_record in remaining_species:
            species_mean = mean(member.fitness for member in species_record.members.values())
            species_record.adjusted_fitness = (species_mean - minimum) / fitness_range
        adjusted = [species_record.adjusted_fitness for species_record in remaining_species]
        previous_sizes = [len(species_record.members) for species_record in remaining_species]
        minimum_size = max(self.reproduction_config.min_species_size, self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted, previous_sizes, pop_size, minimum_size)
        spawn_amounts = self._adjust_spawn_exact(spawn_amounts, pop_size, minimum_size)

        new_population = {}
        species.species = {}
        for spawn, species_record in zip(spawn_amounts, remaining_species):
            spawn = max(spawn, self.reproduction_config.elitism)
            old_members = list(species_record.members.items())
            old_members.sort(reverse=True, key=lambda item: (item[1].fitness, item[0]))
            species_record.members = {}
            species.species[species_record.key] = species_record

            for genome_id, genome in old_members[: self.reproduction_config.elitism]:
                new_population[genome_id] = genome
                spawn -= 1
            while spawn > 0:
                spawn -= 1
                parent1_id, parent1 = epsilon_lexicase_select(old_members)
                parent2_id, parent2 = epsilon_lexicase_select(old_members)
                genome_id = next(self.genome_indexer)
                child = config.genome_type(genome_id)
                child.configure_crossover(parent1, parent2, config.genome_config)
                child.mutate(config.genome_config)
                new_population[genome_id] = child
                self.ancestors[genome_id] = (parent1_id, parent2_id)
        archive = getattr(self, "map_elites_archive", None)
        injection_count = int(getattr(self, "archive_injections", 0))
        if archive is not None and injection_count > 0:
            entries = archive.elites()[:injection_count]
            removable = [genome_id for genome_id, genome in new_population.items() if genome.fitness is None]
            random.shuffle(removable)
            for entry, removed_id in zip(entries, removable):
                del new_population[removed_id]
                genome_id = next(self.genome_indexer)
                injected = copy.deepcopy(entry.genome)
                injected.key = genome_id
                injected.fitness = None
                new_population[genome_id] = injected
                self.ancestors[genome_id] = (entry.genome.key, entry.genome.key)
        return new_population
