from __future__ import annotations

import json
import unittest

from pydantic import TypeAdapter, ValidationError

from hermes_sydney_transport.adapters.tfnsw.wire.trip_planner import (
    JourneyLegWire,
    JourneyPayloadWire,
)


class TripPlannerWireContractTests(unittest.TestCase):
    def test_journey_leg_declares_array_shaped_realtime_status(self):
        payload = b'{"journeys":[{"legs":[{"realtimeStatus":["MONITORED"]}]}]}'

        decoded = TypeAdapter(JourneyPayloadWire).validate_json(payload)

        self.assertEqual(
            decoded.journeys[0].legs[0].realtime_status,
            ("MONITORED",),
        )

    def test_journey_leg_accepts_bounded_long_stop_sequences(self):
        payload = {"stopSequence": [{"id": str(index)} for index in range(59)]}

        decoded = JourneyLegWire.model_validate_json(json.dumps(payload))

        self.assertEqual(len(decoded.stop_sequence), 59)
        with self.assertRaises(ValidationError):
            JourneyLegWire.model_validate_json(
                json.dumps(
                    {"stopSequence": [{"id": str(index)} for index in range(101)]}
                )
            )


if __name__ == "__main__":
    unittest.main()
