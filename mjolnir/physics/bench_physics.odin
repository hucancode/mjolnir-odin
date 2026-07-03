package physics

import "base:runtime"
import "core:log"
import "core:math/linalg"
import "core:testing"
import "core:time"

// All benches force enable_parallel=false to keep odin's test runner happy
// (it cannot host threaded code). They cover the serial code paths.

build_serial_world :: proc() -> ^World {
	w := new(World)
	init(w, {0, -9.81, 0}, false)
	GROUND :: f32(20)
	create_static_body(w, {0, -0.5, 0}, linalg.QUATERNIONF32_IDENTITY, BoxCollider{half_extents = {GROUND, 0.5, GROUND}})

	N :: 6
	for x in 0 ..< N {
		for y in 0 ..< 4 {
			for z in 0 ..< N {
				pos := [3]f32{f32(x - N / 2) * 1.6, 1.2 + f32(y) * 1.6, f32(z - N / 2) * 1.6}
				h := create_dynamic_body(w, pos, linalg.QUATERNIONF32_IDENTITY, 1.0, BoxCollider{half_extents = {0.5, 0.5, 0.5}})
				if body, ok := get_dynamic_body(w, h); ok {
					set_box_inertia(body, {0.5, 0.5, 0.5})
				}
			}
		}
	}
	return w
}

bench_world_state :: struct {
	world:        ^World,
	ccd_handled:  []bool,
	bodies_tested: int,
}

setup_settled_world :: proc(opts: ^time.Benchmark_Options, _: runtime.Allocator) -> time.Benchmark_Error {
	s := cast(^bench_world_state)opts.user_data
	s.world = build_serial_world()
	for _ in 0 ..< 60 do step(s.world, 1.0 / 60.0)
	return .Okay
}

teardown_world :: proc(opts: ^time.Benchmark_Options, _: runtime.Allocator) -> time.Benchmark_Error {
	s := cast(^bench_world_state)opts.user_data
	if len(s.ccd_handled) > 0 do delete(s.ccd_handled)
	shutdown(s.world)
	free(s.world)
	return .Okay
}

@(test)
bench_step_serial :: proc(t: ^testing.T) {
	state: bench_world_state
	opts := time.Benchmark_Options {
		rounds    = 60,
		setup     = setup_settled_world,
		bench     = proc(opts: ^time.Benchmark_Options, _: runtime.Allocator) -> time.Benchmark_Error {
			s := cast(^bench_world_state)opts.user_data
			for _ in 0 ..< opts.rounds do step(s.world, 1.0 / 60.0)
			opts.count = opts.rounds
			return .Okay
		},
		teardown  = teardown_world,
		user_data = &state,
	}
	err := time.benchmark(&opts)
	testing.expect(t, err == .Okay, "bench failed")
	log.infof("step (serial) %d frames in %v, %.0f fps", opts.rounds, opts.duration, opts.rounds_per_second)
}

@(test)
bench_sequential_bvh_refit :: proc(t: ^testing.T) {
	state: bench_world_state
	opts := time.Benchmark_Options {
		rounds    = 5000,
		setup     = setup_settled_world,
		bench     = proc(opts: ^time.Benchmark_Options, _: runtime.Allocator) -> time.Benchmark_Error {
			s := cast(^bench_world_state)opts.user_data
			for _ in 0 ..< opts.rounds do sequential_bvh_refit(s.world)
			opts.count = opts.rounds
			return .Okay
		},
		teardown  = teardown_world,
		user_data = &state,
	}
	err := time.benchmark(&opts)
	testing.expect(t, err == .Okay, "bench failed")
	log.infof("sequential_bvh_refit %d rounds in %v, %.0f rounds/sec", opts.rounds, opts.duration, opts.rounds_per_second)
}

@(test)
bench_sequential_broadphase :: proc(t: ^testing.T) {
	state: bench_world_state
	opts := time.Benchmark_Options {
		rounds    = 2000,
		setup     = setup_settled_world,
		bench     = proc(opts: ^time.Benchmark_Options, _: runtime.Allocator) -> time.Benchmark_Error {
			s := cast(^bench_world_state)opts.user_data
			for _ in 0 ..< opts.rounds {
				clear(&s.world.dynamic_contacts)
				clear(&s.world.static_contacts)
				sequential_collision_detection_traversal(s.world)
			}
			opts.count = opts.rounds
			return .Okay
		},
		teardown  = teardown_world,
		user_data = &state,
	}
	err := time.benchmark(&opts)
	testing.expect(t, err == .Okay, "bench failed")
	pairs := len(state.world.dynamic_contacts) + len(state.world.static_contacts)
	log.infof("sequential_collision_detection_traversal %d rounds in %v, %.0f rounds/sec (pairs/round=%d)",
		opts.rounds, opts.duration, opts.rounds_per_second, pairs)
}

@(test)
bench_sequential_prepare_contacts :: proc(t: ^testing.T) {
	state: bench_world_state
	opts := time.Benchmark_Options {
		rounds    = 5000,
		setup     = proc(opts: ^time.Benchmark_Options, _: runtime.Allocator) -> time.Benchmark_Error {
			setup_settled_world(opts, context.allocator) or_return
			s := cast(^bench_world_state)opts.user_data
			clear(&s.world.dynamic_contacts)
			clear(&s.world.static_contacts)
			sequential_collision_detection_traversal(s.world)
			return .Okay
		},
		bench     = proc(opts: ^time.Benchmark_Options, _: runtime.Allocator) -> time.Benchmark_Error {
			s := cast(^bench_world_state)opts.user_data
			for _ in 0 ..< opts.rounds do sequential_prepare_contacts(s.world, 1.0 / 60.0)
			opts.count = opts.rounds
			return .Okay
		},
		teardown  = teardown_world,
		user_data = &state,
	}
	err := time.benchmark(&opts)
	testing.expect(t, err == .Okay, "bench failed")
	contacts := len(state.world.dynamic_contacts) + len(state.world.static_contacts)
	log.infof("sequential_prepare_contacts %d rounds in %v, %.0f rounds/sec (contacts=%d)",
		opts.rounds, opts.duration, opts.rounds_per_second, contacts)
}

@(test)
bench_sequential_ccd :: proc(t: ^testing.T) {
	state: bench_world_state
	opts := time.Benchmark_Options {
		rounds    = 1000,
		setup     = proc(opts: ^time.Benchmark_Options, _: runtime.Allocator) -> time.Benchmark_Error {
			s := cast(^bench_world_state)opts.user_data
			s.world = build_serial_world()
			for &entry in s.world.bodies.entries {
				if entry.active do entry.item.velocity = {0, -8, 0}
			}
			for _ in 0 ..< 3 do step(s.world, 1.0 / 60.0)
			s.ccd_handled = make([]bool, len(s.world.bodies.entries))
			return .Okay
		},
		bench     = proc(opts: ^time.Benchmark_Options, _: runtime.Allocator) -> time.Benchmark_Error {
			s := cast(^bench_world_state)opts.user_data
			for _ in 0 ..< opts.rounds {
				for i in 0 ..< len(s.ccd_handled) do s.ccd_handled[i] = false
				s.bodies_tested, _ = sequential_ccd(s.world, 1.0 / 60.0, s.ccd_handled)
			}
			opts.count = opts.rounds
			return .Okay
		},
		teardown  = teardown_world,
		user_data = &state,
	}
	err := time.benchmark(&opts)
	testing.expect(t, err == .Okay, "bench failed")
	log.infof("sequential_ccd %d rounds in %v, %.0f rounds/sec (bodies_tested=%d)",
		opts.rounds, opts.duration, opts.rounds_per_second, state.bodies_tested)
}

bench_pair_state :: struct {
	a_box: BoxCollider,
	b_box: BoxCollider,
	a_sph: SphereCollider,
	b_sph: SphereCollider,
	pa:    [3]f32,
	pb:    [3]f32,
	q:     quaternion128,
}

@(test)
bench_narrowphase_box_box :: proc(t: ^testing.T) {
	state := bench_pair_state {
		a_box = {half_extents = {1, 1, 1}}, b_box = {half_extents = {1, 1, 1}},
		pa = {0, 0, 0}, pb = {1.5, 0.3, 0.5}, q = linalg.QUATERNIONF32_IDENTITY,
	}
	opts := time.Benchmark_Options {
		rounds = 200_000, user_data = &state,
		bench = proc(opts: ^time.Benchmark_Options, _: runtime.Allocator) -> time.Benchmark_Error {
			s := cast(^bench_pair_state)opts.user_data
			for _ in 0 ..< opts.rounds do _, _ = collide_boxes(s.pa, s.q, s.a_box, s.pb, s.q, s.b_box, 0)
			opts.count = opts.rounds
			return .Okay
		},
	}
	err := time.benchmark(&opts)
	testing.expect(t, err == .Okay, "bench failed")
	log.infof("collide_boxes %d rounds in %v (%d ns/op)",
		opts.rounds, opts.duration, time.duration_nanoseconds(opts.duration) / i64(opts.rounds))
}

@(test)
bench_narrowphase_sphere_sphere :: proc(t: ^testing.T) {
	state := bench_pair_state {
		a_sph = {radius = 1}, b_sph = {radius = 1},
		pa = {0, 0, 0}, pb = {1.5, 0.3, 0.5}, q = linalg.QUATERNIONF32_IDENTITY,
	}
	opts := time.Benchmark_Options {
		rounds = 200_000, user_data = &state,
		bench = proc(opts: ^time.Benchmark_Options, _: runtime.Allocator) -> time.Benchmark_Error {
			s := cast(^bench_pair_state)opts.user_data
			for _ in 0 ..< opts.rounds do _, _, _, _ = test_sphere_sphere(s.pa, s.a_sph, s.pb, s.b_sph)
			opts.count = opts.rounds
			return .Okay
		},
	}
	err := time.benchmark(&opts)
	testing.expect(t, err == .Okay, "bench failed")
	log.infof("test_sphere_sphere %d rounds in %v (%d ns/op)",
		opts.rounds, opts.duration, time.duration_nanoseconds(opts.duration) / i64(opts.rounds))
}

@(test)
bench_narrowphase_box_sphere :: proc(t: ^testing.T) {
	state := bench_pair_state {
		a_box = {half_extents = {1, 1, 1}}, b_sph = {radius = 1},
		pa = {0, 0, 0}, pb = {1.5, 0.3, 0.5}, q = linalg.QUATERNIONF32_IDENTITY,
	}
	opts := time.Benchmark_Options {
		rounds = 200_000, user_data = &state,
		bench = proc(opts: ^time.Benchmark_Options, _: runtime.Allocator) -> time.Benchmark_Error {
			s := cast(^bench_pair_state)opts.user_data
			for _ in 0 ..< opts.rounds do _, _, _, _ = test_box_sphere(s.pa, s.q, s.a_box, s.pb, s.b_sph)
			opts.count = opts.rounds
			return .Okay
		},
	}
	err := time.benchmark(&opts)
	testing.expect(t, err == .Okay, "bench failed")
	log.infof("test_box_sphere %d rounds in %v (%d ns/op)",
		opts.rounds, opts.duration, time.duration_nanoseconds(opts.duration) / i64(opts.rounds))
}

bench_swept_state :: struct {
	a_he, b_he:      [3]f32,
	r_a, r_b:        f32,
	pa, va, pb:      [3]f32,
	box_min, box_max: [3]f32,
}

@(test)
bench_swept_sphere_sphere :: proc(t: ^testing.T) {
	state := bench_swept_state{r_a = 0.5, r_b = 0.5, pa = {0, 5, 0}, va = {0, -10, 0}, pb = {0.2, 0, 0}}
	opts := time.Benchmark_Options {
		rounds = 200_000, user_data = &state,
		bench = proc(opts: ^time.Benchmark_Options, _: runtime.Allocator) -> time.Benchmark_Error {
			s := cast(^bench_swept_state)opts.user_data
			for _ in 0 ..< opts.rounds do _ = swept_sphere_sphere(s.pa, s.pb, s.r_a, s.r_b, s.va)
			opts.count = opts.rounds
			return .Okay
		},
	}
	err := time.benchmark(&opts)
	testing.expect(t, err == .Okay, "bench failed")
	log.infof("swept_sphere_sphere %d rounds in %v (%d ns/op)",
		opts.rounds, opts.duration, time.duration_nanoseconds(opts.duration) / i64(opts.rounds))
}

@(test)
bench_swept_sphere_box :: proc(t: ^testing.T) {
	pb := [3]f32{0.2, 0, 0}
	state := bench_swept_state{
		r_a = 0.5, pa = {0, 5, 0}, va = {0, -10, 0}, pb = pb,
		box_min = pb - {1, 1, 1}, box_max = pb + {1, 1, 1},
	}
	opts := time.Benchmark_Options {
		rounds = 200_000, user_data = &state,
		bench = proc(opts: ^time.Benchmark_Options, _: runtime.Allocator) -> time.Benchmark_Error {
			s := cast(^bench_swept_state)opts.user_data
			for _ in 0 ..< opts.rounds do _ = swept_sphere_box(s.pa, s.r_a, s.va, s.box_min, s.box_max)
			opts.count = opts.rounds
			return .Okay
		},
	}
	err := time.benchmark(&opts)
	testing.expect(t, err == .Okay, "bench failed")
	log.infof("swept_sphere_box %d rounds in %v (%d ns/op)",
		opts.rounds, opts.duration, time.duration_nanoseconds(opts.duration) / i64(opts.rounds))
}

@(test)
bench_swept_box_box :: proc(t: ^testing.T) {
	state := bench_swept_state{
		a_he = {1, 1, 1}, b_he = {1, 1, 1},
		pa = {0, 5, 0}, va = {0, -10, 0}, pb = {0.2, 0, 0},
	}
	opts := time.Benchmark_Options {
		rounds = 200_000, user_data = &state,
		bench = proc(opts: ^time.Benchmark_Options, _: runtime.Allocator) -> time.Benchmark_Error {
			s := cast(^bench_swept_state)opts.user_data
			for _ in 0 ..< opts.rounds do _ = swept_box_box(s.pa, s.pb, s.a_he, s.b_he, s.va)
			opts.count = opts.rounds
			return .Okay
		},
	}
	err := time.benchmark(&opts)
	testing.expect(t, err == .Okay, "bench failed")
	log.infof("swept_box_box %d rounds in %v (%d ns/op)",
		opts.rounds, opts.duration, time.duration_nanoseconds(opts.duration) / i64(opts.rounds))
}
