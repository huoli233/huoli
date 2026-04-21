import time


def diagnose_proactive_system():
    issues = []
    status = []
    try:
        from src.chat.proactive.schedule_strategy import get_unified_scheduler
        scheduler = get_unified_scheduler()
        if scheduler:
            stats = scheduler.get_statistics()
            status.append(f"统一调度器已初始化: {stats}")
        else:
            issues.append("统一调度器未初始化 (get_unified_scheduler返回None)")
    except Exception as e:
        issues.append(f"统一调度器导入失败: {e}")
    try:
        from src.chat.proactive.silence_detector import get_silence_detector
        detector = get_silence_detector()
        if detector:
            stats = detector.get_statistics()
            status.append(f"沉默检测器已初始化: {stats}")
        else:
            issues.append("沉默检测器未初始化")
    except Exception as e:
        issues.append(f"沉默检测器导入失败: {e}")
    try:
        from src.chat.proactive.think_scheduler import get_think_scheduler
        think_scheduler = get_think_scheduler()
        if think_scheduler:
            stats = think_scheduler.get_statistics()
            active = think_scheduler._is_active
            status.append(f"思考调度器已初始化: active={active}, stats={stats}")
            if not active:
                issues.append("思考调度器未激活 (_is_active=False)，任务不会被执行")
        else:
            issues.append("思考调度器未初始化")
    except Exception as e:
        issues.append(f"思考调度器导入失败: {e}")
    try:
        from src.chat.proactive.think_scheduler import get_think_scheduler, ThinkCategory
        scheduler = get_think_scheduler()
        if scheduler:
            handlers = list(scheduler._handlers.keys())
            if ThinkCategory.SPONTANEOUS_INIT in scheduler._handlers:
                status.append("主动发起任务处理器已注册")
            else:
                issues.append("SPONTANEOUS_INIT 处理器未注册，主动发起任务无法执行")
            status.append(f"已注册的处理器: {[h.value for h in handlers]}")
    except Exception as e:
        issues.append(f"检查handler注册失败: {e}")
    try:
        from src.chat.proactive.message_integration import get_proactive_integration
        proactive = get_proactive_integration()
        if proactive:
            initialized = proactive._initialized
            enabled = proactive._enabled
            has_scheduler = proactive._unified_scheduler is not None
            status.append(f"主动思考集成器: initialized={initialized}, enabled={enabled}, has_scheduler={has_scheduler}")
            if not has_scheduler:
                issues.append("主动思考集成器的_unified_scheduler为None")
        else:
            issues.append("主动思考集成器未初始化")
    except Exception as e:
        issues.append(f"主动思考集成器导入失败: {e}")
    try:
        from src.chat.proactive.engagement_energy import get_engagement_energy_manager
        energy_mgr = get_engagement_energy_manager()
        if energy_mgr:
            status.append("能量管理器已初始化")
        else:
            issues.append("能量管理器未初始化")
    except Exception as e:
        issues.append(f"能量管理器导入失败: {e}")
    try:
        from src.chat.proactive.proactive_core import get_proactive_core
        core = get_proactive_core()
        if core:
            stats = core.get_statistics()
            status.append(f"主动行为核心: {stats}")
        else:
            issues.append("主动行为核心未初始化")
    except Exception as e:
        issues.append(f"主动行为核心导入失败: {e}")
    print("\n" + "=" * 60)
    print("主动回复系统诊断报告")
    print("=" * 60)
    print("\n系统状态:")
    for s in status:
        print(f"  {s}")
    if issues:
        print("\n发现的问题:")
        for i in issues:
            print(f"  {i}")
    else:
        print("\n未发现明显问题")
    print("\n" + "=" * 60)
    return issues, status


def check_quiet_time():
    try:
        from src.chat.proactive.schedule_strategy import is_quiet_time
        quiet_hours = "23-7"
        is_quiet = is_quiet_time(quiet_hours)
        print(f"当前是否在免打扰时段({quiet_hours}): {is_quiet}")
        return is_quiet
    except Exception as e:
        print(f"检查免打扰时段失败: {e}")
        return None


if __name__ == "__main__":
    print("开始诊断主动回复系统...")
    diagnose_proactive_system()
    print()
    check_quiet_time()
